import os
import pickle
import sys
from dotenv import load_dotenv
from chromadb.utils import embedding_functions

# Import search queries
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils.search_queries import GAMER_TYPE_QUERIES, GENERAL_QUERY

CACHE_FILE = os.path.join(parent_dir, "datasets", "query_cache.pkl")

class QueryCacheManager:
    def __init__(self):
        self.cache = {}
        self.embedding_fn = self._get_embedding_function()
        self._load_cache()

    def _get_embedding_function(self):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file.")
            
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name="text-embedding-3-small"
        )

    def _load_cache(self):
        if os.path.exists(CACHE_FILE):
            print(f"📖 Loading query cache from {CACHE_FILE}...")
            try:
                with open(CACHE_FILE, 'rb') as f:
                    self.cache = pickle.load(f)
                print(f"✅ Loaded {len(self.cache)} embeddings.")
            except Exception as e:
                print(f"⚠️ Failed to load cache: {e}. Starting fresh.")
                self.cache = {}
        else:
            print("🆕 No cache found. It will be created.")
            self.cache = {}

    def _save_cache(self):
        try:
            with open(CACHE_FILE, 'wb') as f:
                pickle.dump(self.cache, f)
            print(f"💾 Cache saved to {CACHE_FILE} ({len(self.cache)} embeddings).")
        except Exception as e:
            print(f"⚠️ Failed to save cache: {e}")

    def get_embedding(self, query_text):
        if query_text in self.cache:
            return self.cache[query_text]
        
        # If not in cache, generate it
        print(f"🔂 Embedding new query: '{query_text}'")
        # OpenAIEmbeddingFunction calls return a list of embeddings
        embedding = self.embedding_fn([query_text])[0]
        self.cache[query_text] = embedding
        self._save_cache() # Save immediately or strategy? Save immediately for safety.
        return embedding

    def precompute_all(self):
        """Pre-compute embeddings for all known static queries."""
        all_queries = set()
        all_queries.add(GENERAL_QUERY)
        for queries in GAMER_TYPE_QUERIES.values():
            all_queries.update(queries)
        
        missing = [q for q in all_queries if q not in self.cache]
        
        if not missing:
            print("✅ All queries are already cached.")
            return

        print(f"🔄 Pre-computing {len(missing)} missing embeddings...")
        
        # Batch process could be better, but simple loop is fine for <100 queries
        # Actually batching is better for API
        batch_size = 20
        all_missing = list(missing)
        
        for i in range(0, len(all_missing), batch_size):
            batch = all_missing[i : i + batch_size]
            print(f"   Processing batch {i//batch_size + 1}...")
            embeddings = self.embedding_fn(batch)
            for q, emb in zip(batch, embeddings):
                self.cache[q] = emb
        
        self._save_cache()
        print("✅ Pre-computation complete.")

# Global instance
_manager = None

def get_query_embedding(query_text):
    global _manager
    if _manager is None:
        _manager = QueryCacheManager()
        # Ensure precompute happens on first load?
        # Or just lazy load. User asked to "caching mechanism".
        # Let's run precompute if cache is empty or small.
        _manager.precompute_all()
        
    return _manager.get_embedding(query_text)

if __name__ == "__main__":
    # Run this script directly to build cache
    print("🛠️ Building Query Cache...")
    cm = QueryCacheManager()
    cm.precompute_all()
