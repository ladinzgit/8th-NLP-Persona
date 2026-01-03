import chromadb
import os
import openai
from chromadb.utils import embedding_functions
import sys
import pickle
import torch

# 프로젝트 루트 경로 추가 (모듈 import를 위해)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.search_queries import GAMER_TYPE_QUERIES, GENERAL_QUERY

# ChromaDB 경로 설정 (Set ChromaDB Path)
CHROMA_DB_PATH = "datasets/chroma_db"
COLLECTION_NAME = "cyberpunk2077_reviews"
CACHE_PATH = "datasets/query_cache.pkl"
# Model Path (Same as build_chroma_db.py)
MODEL_PATH = os.path.join("models", "Qwen3-Embedding-0.6B")

def get_chroma_client():
    """
    ChromaDB 영구 클라이언트(PersisentClient)를 반환합니다.
    Returns:
        chromadb.PersistentClient: ChromaDB 클라이언트 객체
    """
    # 데이터베이스 디렉토리가 존재하는지 확인
    if not os.path.exists(CHROMA_DB_PATH):
        raise FileNotFoundError(f"ChromaDB 경로를 찾을 수 없습니다: {CHROMA_DB_PATH}")
    
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client

class CustomEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self, model_path):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Please install sentence_transformers: pip install sentence-transformers")
        
        print(f"Loading embedding model from: {model_path}")
        # trust_remote_code=True might be needed for some Qwen models
        self.model = SentenceTransformer(model_path, trust_remote_code=True, device="cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, input: list) -> list:
        # Generate embeddings
        embeddings = self.model.encode(input, convert_to_tensor=False).tolist()
        return embeddings

class RAGRetriever:
    def __init__(self):
        """
        RAGRetriever 초기화
        ChromaDB 클라이언트와 컬렉션을 로드하고, 쿼리 캐시를 로드합니다.
        """
        self.client = get_chroma_client()
        
        # 1. Load Query Cache
        self.query_cache = {}
        if os.path.exists(CACHE_PATH):
            try:
                with open(CACHE_PATH, 'rb') as f:
                    self.query_cache = pickle.load(f)
                print(f"Query cache loaded: {len(self.query_cache)} entries.")
            except Exception as e:
                print(f"Warning: Failed to load query cache: {e}")
                
        # 2. Setup Embedding Function (Lazy Load Logic)
        # If we have cache, we might not need to load model immediately.
        # But we need an EF for get_collection usually.
        # However, we can use a Dummy EF if we always provide embeddings, 
        # OR we assume cache covers everything.
        # User requirement: "If not in cache, assume model loading."
        
        self.ef = None # Lazy
        self.model_loaded = False
        
        # To get collection, we need an EF passed? 
        # Actually Chroma allows getting collection without EF if we query with embeddings?
        # Let's try to get collection with a simple Default EF first to avoid loading heavy model just for handle.
        # But if the collection was created with specific EF, ideally we match it.
        # BUT since we inject embeddings manually, we can pass a dummy or lightweight EF (like MiniLM) just for object creation,
        # provided we ALWAYS use query_embeddings=[...]
        
        # However, to be safe and follow user instruction "use same model if not cached",
        # let's define a wrapper that loads on demand.
        
        self.lazy_ef = self._lazy_embedding_function
        
        try:
            self.collection = self.client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=self.lazy_ef # Pass lazy function wrapper
            )
            print(f"Collection '{COLLECTION_NAME}' loaded successfully.")
        except Exception as e:
            raise ValueError(f"컬렉션 '{COLLECTION_NAME}'을 로드하는 중 오류가 발생했습니다: {e}")

    def _lazy_embedding_function(self, input: list) -> list:
        """
        Wrapper to load model only when needed.
        """
        if not self.model_loaded:
            print("Cache Miss or Manual Embedding required. Loading Qwen model...")
            if os.path.exists(MODEL_PATH):
                self.real_ef = CustomEmbeddingFunction(MODEL_PATH)
            else:
                print("Warning: Local model not found. Using default MiniLM.")
                self.real_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
            self.model_loaded = True
            
        return self.real_ef(input)

    def retrieve_reviews(self, query_text, current_date, top_k=5):
        """
        주어진 쿼리와 날짜를 기준으로 관련 리뷰를 검색합니다.
        """
        # 날짜 포맷 변환 (YYYYMMDD)
        try:
            date_int = int(current_date.replace("-", ""))
        except ValueError:
            print(f"Warning: Invalid date format {current_date}. Using current timestamp.")
            date_int = 20250101 
            
        where_filter = {"date": {"$lte": date_int}}
        
        # --- Caching Logic ---
        query_embeddings = None
        
        if query_text in self.query_cache:
            # Cache Hit
            # print(f"Cache Hit for: '{query_text}'")
            query_embeddings = [self.query_cache[query_text]]
        else:
            # Cache Miss -> Trigger Lazy Load via _lazy_embedding_function via call inside query?
            # Or manually call it.
            # If we pass query_texts=[query_text], Chroma calls self.lazy_ef([query_text]).
            # This handles it automatically!
            # BUT to be explicit about cache usage:
            # If we have embeddings, pass query_embeddings.
            pass

        if query_embeddings:
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=top_k,
                where=where_filter
            )
        else:
            # Fallback to EF (Model Load)
            results = self.collection.query(
                query_texts=[query_text],
                n_results=top_k,
                where=where_filter
            )
        
        if results['documents'] and results['documents'][0]:
            # Formatter
            formatted_results = []
            docs = results['documents'][0]
            metas = results['metadatas'][0]
            
            for doc, meta in zip(docs, metas):
                date_int = meta.get('date', 0)
                date_str = str(date_int)
                if len(date_str) == 8:
                    date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
                
                review_snippet = doc[:400]
                formatted_results.append(f"- [{date_str}] {review_snippet}...")
                
            return formatted_results
        else:
            return []

if __name__ == "__main__":
    # 테스트 코드 (Test Code)
    try:
        retriever = RAGRetriever()
        
        # 테스트 쿼리
        test_query = "Cyberpunk 2077 bugs and glitches"
        test_date = "2023-12-31"
        
        print(f"\n--- Testing Retrieval for date: {test_date} ---")
        reviews = retriever.retrieve_reviews(test_query, test_date, top_k=3)
        
        for i, review in enumerate(reviews):
            print(f"\nReview {i+1}:")
            print(review[:200] + "...") # 앞부분만 출력
            
    except Exception as e:
        print(f"Error during test: {e}")
