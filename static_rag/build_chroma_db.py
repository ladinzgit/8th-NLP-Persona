import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime
import torch
import pickle
from langdetect import detect, LangDetectException
import os
import sys
import argparse
import gc

# OOM 방지를 위한 설정 (메모리 단편화 감소)
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 프로젝트 루트 경로 추가 (utils 모듈 import를 위함)
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.search_queries import GAMER_TYPE_QUERIES, GENERAL_QUERY

# 설정 (Configuration)
CSV_PATH = os.path.join("datasets", "reviews", "Cyberpunk_2077_Steam_Reviews.csv")
DB_PATH = os.path.join("datasets", "chroma_db")
COLLECTION_NAME = "cyberpunk2077_reviews"
CACHE_PATH = os.path.join("datasets", "query_cache.pkl")

# 로컬 모델 경로 (프로젝트 루트 기준 상대 경로 또는 절대 경로)
# 사용자가 지정한 모델: models/Qwen3-Embedding-0.6B
MODEL_PATH = os.path.join("models", "Qwen3-Embedding-0.6B")
MODEL_NAME = "Qwen3-Embedding-0.6B" # 메타데이터 또는 폴백(fallback)용

def process_reviews(csv_path, limit=None):
    """
    CSV 파일을 읽고 전처리합니다.
    유효한 리뷰를 필터링하고 날짜 형식을 변환합니다.
    """
    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
        sys.exit(1)
        
    print(f"Total rows: {len(df)}")
    
    if limit:
        print(f"Limiting to first {limit} rows for testing...")
        df = df.head(limit)
    
    # 'Review' 컬럼 존재 여부 확인
    if 'Review' not in df.columns:
        print(f"Error: 'Review' column not found. Columns: {df.columns.tolist()}")
        sys.exit(1)

    # 빈 리뷰 제거
    df = df.dropna(subset=['Review'])
    print(f"Rows after dropping empty reviews: {len(df)}")
    
    # 언어 필터링 (영어 리뷰만 추출)
    print("Filtering for English reviews (this may take a while)...")
    
    def is_english(text):
        try:
            return detect(str(text)) == 'en'
        except LangDetectException:
            return False
            
    # 필터링 적용
    # 대량 데이터 처리 시 swifter 등을 고려할 수 있으나 의존성 최소화를 위해 apply 사용
    df['is_english'] = df['Review'].apply(is_english)
    
    # 필터링 결과 로그
    english_count = df['is_english'].sum()
    print(f"English reviews found: {english_count} / {len(df)}")
    
    df = df[df['is_english']]
    
    return df

def parse_date_to_int(date_str):
    """
    날짜 문자열을 YYYYMMDD 정수형으로 변환합니다.
    지원 형식: 'M/D/YYYY', 'YYYY-MM-DD', 'DD-MM-YYYY', 'Month D, YYYY' 등
    """
    if pd.isna(date_str):
        return None
        
    formats = [
        '%m/%d/%Y',      # 12/9/2020
        '%Y-%m-%d',      # 2020-12-09
        '%d-%m-%Y',      # 09-12-2020
        '%B %d, %Y'      # December 9, 2020
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(str(date_str).strip(), fmt)
            return int(dt.strftime('%Y%m%d'))
        except ValueError:
            continue
            
    return None

def generate_query_cache(embedding_function):
    """
    search_queries.py에 정의된 모든 쿼리에 대해 임베딩을 생성하고 캐시(pkl)로 저장합니다.
    """
    print(f"\n--- Generating Query Cache ---")
    
    # 1. 모든 쿼리 수집
    all_queries = set()
    all_queries.add(GENERAL_QUERY)
    
    for queries in GAMER_TYPE_QUERIES.values():
        for q in queries:
            all_queries.add(q)
            
    unique_queries = list(all_queries)
    print(f"Total unique queries to cache: {len(unique_queries)}")
    
    # 2. 임베딩 생성 (기존 캐시가 있어도 모델 일관성을 위해 덮어씁니다)
    start_time = datetime.now()
    embeddings = embedding_function(unique_queries)
    end_time = datetime.now()
    print(f"Embedding generation took: {end_time - start_time}")
    
    # 3. 딕셔너리로 저장
    cache_data = {}
    for q, emb in zip(unique_queries, embeddings):
        cache_data[q] = emb
        
    try:
        with open(CACHE_PATH, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"✅ Query cache saved to: {CACHE_PATH}")
    except Exception as e:
        print(f"❌ Failed to save cache: {e}")

class CustomEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self, model_path):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Please install sentence_transformers: pip install sentence-transformers")
        
        print(f"Loading embedding model from: {model_path}")
        # 일부 Qwen 모델은 trust_remote_code=True가 필요할 수 있음
        self.model = SentenceTransformer(model_path, trust_remote_code=True, device="cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, input: list) -> list:
        # 임베딩 생성 (텐서 변환 없이 리스트로 반환)
        embeddings = self.model.encode(input, convert_to_tensor=False).tolist()
        return embeddings

def build_chroma_db(test_mode=False, batch_size=32):
    # ChromaDB 클라이언트 초기화
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # 임베딩 함수 설정 (로컬 Qwen 모델 우선 사용, 없으면 MiniLM 폴백)
    if os.path.exists(MODEL_PATH) and os.listdir(MODEL_PATH):
        print(f"Found local model at {MODEL_PATH}, using CustomEmbeddingFunction.")
        ef = CustomEmbeddingFunction(model_path=MODEL_PATH)
    else:
        print(f"Warning: Local model not found at {MODEL_PATH}. Falling back to default {MODEL_NAME}.")
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    # --- DB 확인 로직 ---
    # 사용자의 요청에 따라 기존 DB가 있어도 무조건 삭제하고 다시 만듭니다.
    print(f"Force rebuilding collection '{COLLECTION_NAME}'...")
    should_rebuild_db = True
    
    # if db_exists: ... 로직 제거됨

    if should_rebuild_db:
        # 기존 컬렉션 삭제 후 재생성
        try:
            client.delete_collection(name=COLLECTION_NAME)
            print(f"Deleted existing collection: {COLLECTION_NAME}")
        except Exception as e:
            pass # 컬렉션이 없는 경우 무시

        collection = client.create_collection(name=COLLECTION_NAME, embedding_function=ef)
        
        df = process_reviews(CSV_PATH, limit=5000 if test_mode else None)
        
        if test_mode:
            print("Test mode: Processing only first 5000 records.")
            # df = df.head(5000) # process_reviews에서 이미 처리됨
        
        # batch_size는 인자로 전달받음 (기본값: 32)
        total_docs = len(df)
        
        documents = []
        metadatas = []
        ids = []
        
        print("Starting ingestion...")
        
        for i, (idx, row) in enumerate(df.iterrows()):
            review_text = row['Review']
            date_val = row.get('Date Posted')
            
            # 날짜 파싱
            date_int = parse_date_to_int(date_val)
            if not date_int:
                continue
                
            # 메타데이터 구성
            # Rating: 'Recommended' -> True, 'Not Recommended' -> False
            rating_str = str(row.get('Rating', '')).lower()
            is_positive = 'recommended' in rating_str and 'not' not in rating_str
            
            # 플레이 타임 (Playtime) 처리
            playtime = 0.0
            try:
                if 'Playtime' in row:
                    pt_str = str(row['Playtime']).replace('hours', '').strip()
                    playtime = float(pt_str)
            except:
                pass
                
            metadata = {
                "date": date_int,
                "rating": rating_str,
                "voted_up": is_positive,
                "playtime": playtime,
                "source": "steam_new_dataset"
            }
            
            # ID 설정
            doc_id = str(row['ReviewID']) if 'ReviewID' in row else f"rev_{i}"
            
            documents.append(review_text)
            metadatas.append(metadata)
            ids.append(doc_id)
            
            if len(documents) >= batch_size:
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                documents = []
                metadatas = []
                ids = []
                
                # 메모리 정리 (OOM 방지)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                print(f"Processed {i + 1}/{total_docs} reviews...", end='\r')
                
        # 남은 데이터 처리
        if documents:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
        
        print(f"\nIngestion complete. Total documents in collection: {collection.count()}")

        if test_mode:
            verify_insertion(collection)
            
    # --- 쿼리 캐시 생성 ---
    # DB 빌드 스크립트 실행 시 항상 캐시를 갱신합니다.
    generate_query_cache(ef)

def verify_insertion(collection):
    print("\n--- Verification ---")
    print("Querying for 'cyberpunk'...")
    results = collection.query(
        query_texts=["Great open world game"],
        n_results=1
    )
    print("Result:")
    print(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build ChromaDB from new reviews CSV.')
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for ingestion (default: 32)')
    args = parser.parse_args()
    
    build_chroma_db(test_mode=args.test, batch_size=args.batch_size)
