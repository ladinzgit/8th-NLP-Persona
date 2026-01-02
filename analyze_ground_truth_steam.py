import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# 1. 데이터 로드
csv_file = "Cyberpunk_2077_Steam_Reviews.csv" 
print(f"Loading {csv_file}...")

df = pd.read_csv(csv_file, usecols=['Rating', 'Date Posted'])

# 2. 데이터 전처리
print("Processing data...")
# 날짜 변환 (형식에 맞춰 변환, 에러 데이터는 제거)
df['Date Posted'] = pd.to_datetime(df['Date Posted'], format='%m/%d/%Y', errors='coerce')

# Rating 변환 (Recommended=1, Not Recommended=0)
df['Is_Positive'] = df['Rating'].apply(lambda x: 1 if x == 'Recommended' else 0)

# 유효하지 않은 날짜 제거
df = df.dropna(subset=['Date Posted'])

# 3. 일별 긍정 비율 계산 (전체 데이터 사용)
daily_sentiment = df.groupby('Date Posted')['Is_Positive'].mean().reset_index()
daily_sentiment.columns = ['Date', 'Positive_Ratio']
daily_sentiment = daily_sentiment.sort_values('Date')

# 4. 이동 평균(Rolling Average) 계산 (7일 기준)
# 일별 데이터는 너무 들쑥날쑥하므로 추세를 보기 위해 이동평균 사용하였음
daily_sentiment['Smoothed_Ratio'] = daily_sentiment['Positive_Ratio'].rolling(window=7, min_periods=1).mean()

# 5. 그래프 그리기
print("Generating clean graph...")
plt.figure(figsize=(15, 6))
sns.set_theme(style="whitegrid")

# 메인 그래프 (이동 평균선)
plt.plot(daily_sentiment['Date'], daily_sentiment['Smoothed_Ratio'], linestyle='-', linewidth=2, color='#00a8cc', label='7-Day Moving Avg Sentiment')

min_date = daily_sentiment['Date'].min()
max_date = daily_sentiment['Date'].max()
plt.xlim(min_date, max_date)
plt.ylim(0, 1.1)

# 그래프 스타일링
plt.title('Cyberpunk 2077 Steam Review Sentiment Trend', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Positive Review Ratio (0.0 - 1.0)', fontsize=12)

# X축 날짜 포맷 (연도-월)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)

plt.legend(loc='lower right')
plt.tight_layout()

# 7. 저장
output_img = "ground_truth_steam.png"
plt.savefig(output_img, dpi=300)
print(f"Graph saved to '{output_img}'")

daily_sentiment.to_csv("ground_truth_steam.csv", index=False)
print("'ground_truth_steam.csv' 저장 완료")

