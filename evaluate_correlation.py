import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import os

sns.set_theme(style="whitegrid")

def parse_args():
    parser = argparse.ArgumentParser(description="모델 예측과 ground truth(주가, 긍정 리뷰 비율)의 피어슨 상관계수 계산")
    
    # 필수 인자: 모델 결과 파일 경로
    parser.add_argument("--model_csv", type=str, required=True, help="CSV file 경로")
    
    # 필수 인자: 모델 이름 (그래프 라벨용)
    parser.add_argument("--model_name", type=str, required=True, help="모델의 이름 (e.g., 'Team1_Static')")
    
    # 필수 인자: 모델 타입 (static: 1팀용 / dynamic: 2,3팀용)
    parser.add_argument("--type", type=str, choices=['static', 'dynamic'], required=True, help="'static' for Team 1 (시간에 따라 변화없음), 'dynamic' for Team 2/3 (시간에 따라 바뀜)")
    
    # 정답지 경로 (기본값 설정)
    parser.add_argument("--steam_gt", type=str, default="ground_truth_steam.csv", help="Path to Steam GT CSV")
    parser.add_argument("--stock_gt", type=str, default="ground_truth_stock.csv", help="Path to Stock GT CSV")
    
    return parser.parse_args()

def load_ground_truth(path, value_col):
    """정답지 CSV를 로드하고 날짜를 인덱스로 설정"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ground Truth file not found: {path}")
    
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df[['Date', value_col]].sort_values('Date')

def calculate_model_ratio(df_model, model_type):
    """모델 결과 CSV에서 구매 비율(YES Ratio) 추출"""
    
    # YES/NO 대소문자 통일 및 1/0 변환
    df_model['Vote'] = df_model['Decision'].apply(lambda x: 1 if str(x).strip().upper().startswith('YES') else 0)
    
    if model_type == 'static':
        # 1팀: 전체 데이터의 평균을 하나의 상수로 반환
        ratio = df_model['Vote'].mean()
        return ratio, None
        
    elif model_type == 'dynamic':
        # 2,3팀: 'Simulation_Date' 컬럼이 있어야 함 (없으면 에러)
        if 'Simulation_Date' not in df_model.columns:
            # 만약 Date 컬럼 이름이 다르다면 여기서 수정
            if 'Date' in df_model.columns:
                date_col = 'Date'
            else:
                raise ValueError("날짜 컬럼이 없습니다. (e.g., 'Simulation_Date')")
        else:
            date_col = 'Simulation_Date'
            
        df_model[date_col] = pd.to_datetime(df_model[date_col])
        daily_ratio = df_model.groupby(date_col)['Vote'].mean().reset_index()
        daily_ratio.columns = ['Date', 'Purchase_Ratio']
        return None, daily_ratio

def main():
    args = parse_args()
    
    print(f"--- 평가 모델: {args.model_name} ({args.type}) ---")
    
    # 1. 정답지 로드
    steam_gt = load_ground_truth(args.steam_gt, 'Positive_Ratio')
    stock_gt = load_ground_truth(args.stock_gt, 'Stock_Price')
    
    # 2. 모델 결과 로드
    model_df = pd.read_csv(args.model_csv)
    
    # 3. 모델 구매 비율 계산
    static_ratio, dynamic_df = calculate_model_ratio(model_df, args.type)
    
    # 4. 비교를 위한 데이터 병합 (Merge)
    if args.type == 'static': # static
        # Static: Steam 데이터의 날짜만큼 모델 값을 복사해서 채움
        merged_steam = steam_gt.copy()
        merged_steam['Model_Ratio'] = static_ratio
        
        # Stock 데이터와도 병합
        merged_stock = stock_gt.copy()
        merged_stock['Model_Ratio'] = static_ratio
        
        print(f"   [Info] Static 비율 계산 완료: {static_ratio:.4f}")
        
    else: # dynamic
        # 날짜 기준으로 병합 (교집합 날짜만 평가)
        merged_steam = pd.merge(steam_gt, dynamic_df, on='Date', how='inner')
        merged_steam.rename(columns={'Purchase_Ratio': 'Model_Ratio'}, inplace=True)
        
        merged_stock = pd.merge(stock_gt, dynamic_df, on='Date', how='inner')
        merged_stock.rename(columns={'Purchase_Ratio': 'Model_Ratio'}, inplace=True)
        
        print(f"   [Info] {len(merged_steam)} 의 날짜 매치됨")

    # 5. 피어슨 상관계수 계산
    # (데이터 포인트가 2개 미만이면 계산 불가)
    if len(merged_steam) < 2:
        print("Error: GT와 겹치는 날짜가 부족합니다.")
        return

    corr_steam, _ = pearsonr(merged_steam['Model_Ratio'], merged_steam['Positive_Ratio'])
    corr_stock, _ = pearsonr(merged_stock['Model_Ratio'], merged_stock['Stock_Price'])
    
    print("\n" + "="*40)
    print(f"[{args.model_name}]의 평과 결과")
    print("="*40)
    print(f"1. Correlation with Steam Sentiment:  {corr_steam:.4f}")
    print(f"2. Correlation with Stock Price:      {corr_stock:.4f}")
    
    # Static 모델은 보통 NaN(변동 없음)이나 0이 나올 수 있음 -> 경고 메시지 처리
    if np.isnan(corr_steam):
        print("   (Note: Correlation is NaN because the model output has zero variance.)")
    print("="*40 + "\n")

    # 6. 시각화 및 저장
    plt.figure(figsize=(12, 10))
    
    # 서브플롯 1: Steam 비교
    plt.subplot(2, 1, 1)
    plt.plot(merged_steam['Date'], merged_steam['Positive_Ratio'], 'b-', label='Steam GT (Sentiment)', alpha=0.6)
    plt.plot(merged_steam['Date'], merged_steam['Model_Ratio'], 'r--o', label=f'{args.model_name} (Prediction)', linewidth=2)
    plt.title(f'Comparison: {args.model_name} vs Steam Sentiment (r={corr_steam:.2f})')
    plt.legend()
    
    # 서브플롯 2: Stock 비교
    plt.subplot(2, 1, 2)
    # 주가는 단위가 다르므로 정규화(0~1)하거나 2축을 써야 보기가 좋음. 여기서는 2축 사용.
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.plot(merged_stock['Date'], merged_stock['Stock_Price'], 'g-', label='Stock Price (GT)', alpha=0.6)
    ax2.plot(merged_stock['Date'], merged_stock['Model_Ratio'], 'r--o', label=f'{args.model_name} (Prediction)', linewidth=2)
    
    ax1.set_ylabel('Stock Price (Currency)')
    ax2.set_ylabel('Purchase Probability (0-1)')
    plt.title(f'Comparison: {args.model_name} vs Stock Price (r={corr_stock:.2f})')
    
    # 저장
    save_path = f"eval_{args.model_name}_graph.png"
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"{save_path}의 경로로 저장 완료!")

if __name__ == "__main__":
    main()

