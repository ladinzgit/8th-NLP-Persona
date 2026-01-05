import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime
import os
import argparse
import sys
from dotenv import load_dotenv
import concurrent.futures

# 설정 (Configuration)
CSV_PATH = os.path.join("datasets", "cyberpunk2077_all_reviews.csv")
DB_PATH = os.path.join("datasets", "chroma_db_new")
COLLECTION_NAME = "cyberpunk2077_reviews"

# Load OpenAI API Key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("⚠️ Error: OPENAI_API_KEY not found in .env file.")
    sys.exit(1)

def process_reviews(csv_path):
    """
    CSV 파일을 읽고 처리합니다.
    유효한 리뷰를 필터링하고 날짜 형식을 변환합니다.
    """
    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
        sys.exit(1)
        
    print(f"Total rows: {len(df)}")
    
    # 'review' 컬럼 존재 여부 확인 (lowercase)
    if 'review' not in df.columns:
        print(f"Error: 'review' column not found. Columns: {df.columns.tolist()}")
        sys.exit(1)

    # 1. English Only Filtering
    if 'language' in df.columns:
        # 'english' only
        original_len = len(df)
        df = df[df['language'].str.lower() == 'english']
        print(f"Filtered non-english reviews: {original_len} -> {len(df)}")
    else:
        print("Warning: 'language' column not found, skipping language filtering.")

    # 빈 리뷰 제거
    df = df.dropna(subset=['review'])
    
    return df



import time
import random

def process_batch(collection, batch_data, batch_idx):
    """
    Process a single batch of data and add to collection.
    Includes retry logic for Rate Limit (429) errors.
    """
    documents, metadatas, ids = batch_data
    max_retries = 5
    base_delay = 2
    
    for attempt in range(max_retries + 1):
        try:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            return len(documents)
        except Exception as e:
            error_msg = str(e).lower()
            if "rate limit" in error_msg or "429" in error_msg:
                if attempt < max_retries:
                    delay = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                    print(f"\n⚠️ Rate limit hit on Batch {batch_idx}. Retrying in {delay:.1f}s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(delay)
                    continue
            
            print(f"\n❌ Batch {batch_idx} failed: {e}")
            return 0

def build_chroma_db(test_mode=False):
    # ChromaDB 클라이언트 초기화
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # 임베딩 함수 설정 (OpenAI text-embedding-3-small 사용)
    print("Using OpenAI embedding model: text-embedding-3-small")
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-embedding-3-small"
    )
    
    # 스키마 초기화를 위해 항상 기존 컬렉션 삭제
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")
    except Exception as e:
        print(f"Collection deletion skipped: {e}")

    collection = client.create_collection(name=COLLECTION_NAME, embedding_function=ef)
    
    df = process_reviews(CSV_PATH)
    
    if test_mode:
        print("Test mode: Processing only first 5000 records.")
        df = df.head(5000)
    
    batch_size = 512
    total_docs = len(df)
    
    print(f"\n{'='*60}")
    print(f"🚀 ChromaDB 구축 시작 (Parallel Mode - Safe)")
    print(f"{'='*60}")
    print(f"📊 총 리뷰 수: {total_docs:,}개")
    print(f"📦 배치 크기: {batch_size}개")
    print(f"💾 DB 경로: {DB_PATH}")
    print(f"📚 컬렉션명: {COLLECTION_NAME}")
    print(f"🤖 Embedding: text-embedding-3-small")
    print(f"{'='*60}\n")
    
    skipped_count = 0
    processed_count = 0
    
    # Prepare all batches first
    batches = []
    current_batch = {'docs': [], 'metas': [], 'ids': []}
    
    print("Preparing batches...")
    for i, (idx, row) in enumerate(df.iterrows()):
        review_text = row['review']
        
        # 날짜 파싱 (Unix Timestamp 사용)
        # timestamp_updated가 우선, 없으면 timestamp_created 차선
        ts = row.get('timestamp_updated')
        if pd.isna(ts):
            ts = row.get('timestamp_created')
            
        try:
            # Unix timestamp to YYYYMMDD int
            dt = datetime.fromtimestamp(int(ts))
            date_int = int(dt.strftime('%Y%m%d'))
        except (ValueError, TypeError):
             skipped_count += 1
             if skipped_count <= 5:
                 print(f"⚠️  Invalid timestamp: {ts} (row {i+1})")
             continue
            
        # 메타데이터 구성
        rating_str = str(row.get('Rating', '')).lower()
        is_positive = 'recommended' in rating_str and 'not' not in rating_str
        
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
        
        doc_id = str(row['ReviewID']) if 'ReviewID' in row else f"rev_{i}"
        
        current_batch['docs'].append(review_text)
        current_batch['metas'].append(metadata)
        current_batch['ids'].append(doc_id)
        
        if len(current_batch['docs']) >= batch_size:
            batches.append((
                current_batch['docs'], 
                current_batch['metas'], 
                current_batch['ids']
            ))
            current_batch = {'docs': [], 'metas': [], 'ids': []}
            
    if current_batch['docs']:
        batches.append((
            current_batch['docs'], 
            current_batch['metas'], 
            current_batch['ids']
        ))
        
    print(f"Total batches to process: {len(batches)}")
    
    # Process batches in parallel
    # Reduced max_workers to avoid Rate Limits
    max_workers = 3 
    total_processed = 0
    
    print(f"Running with {max_workers} threads to respect API limits...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_batch, collection, batch, idx): idx 
                  for idx, batch in enumerate(batches)}
        
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            total_processed += res
            print(f"✅ Processed: {total_processed:,}/{total_docs:,} reviews ({(total_processed/total_docs)*100:.1f}%)", end='\r')
            
    print(f"\n{'='*60}")
    print(f"✅ ChromaDB 구축 완료!")
    print(f"{'='*60}")
    print(f"📊 처리 통계:")
    print(f"   - 총 리뷰 수: {total_docs:,}개")
    print(f"   - 성공적으로 저장: {total_processed:,}개")
    print(f"   - 건너뛴 리뷰 (날짜 파싱 실패): {skipped_count:,}개")
    print(f"   - 최종 저장된 문서 수: {collection.count():,}개")
    print(f"{'='*60}\n")

    if test_mode:
        verify_insertion(collection)

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
    args = parser.parse_args()
    
    build_chroma_db(test_mode=args.test)
