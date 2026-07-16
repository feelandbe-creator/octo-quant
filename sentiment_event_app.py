import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import re

# 페이지 기본 설정
st.set_page_config(page_title="Event-Driven Sentiment Alpha", layout="wide")
st.title("🔥 Real-Time NLP Event-Driven Sentiment Alpha (V1)")
st.caption(f"시스템 가동 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (KST)")

# --- 1. 금융 특화 감성 사전 (Loughran-McDonald 기반 초경량 커스텀 사전) ---
FIN_LEXICON = {
    # 강력 호재 (Positive)
    "beat": 1.0, "surprise": 0.8, "growth": 0.7, "surge": 0.9, "profit": 0.8,
    "recovery": 0.6, "rebound": 0.6, "upgrade": 0.8, "bullish": 0.9, "record-high": 1.0,
    "acquisition": 0.5, "dividend": 0.6, "outperform": 0.8, "breakthrough": 0.9,
    # 강력 악재 (Negative)
    "miss": -1.0, "decline": -0.7, "slump": -0.9, "loss": -0.8, "drop": -0.6,
    "downgrade": -0.8, "bearish": -0.9, "deficit": -0.8, "bankruptcy": -1.0,
    "lawsuit": -0.5, "fine": -0.4, "investigation": -0.6, "tariff": -0.7, "inflation": -0.5
}

# --- 2. NLP 엔진 함수 ---
def analyze_sentiment(text):
    text = text.lower()
    # 특수문자 제거 및 단어 토큰화
    words = re.findall(r'\b\w+\b', text)
    
    pos_score = 0
    neg_score = 0
    matched_words = []
    
    for word in words:
        if word in FIN_LEXICON:
            val = FIN_LEXICON[word]
            matched_words.append(f"{word}({val})")
            if val > 0:
                pos_score += val
            else:
                neg_score += abs(val)
                
    total = pos_score + neg_score
    # 감성 점수 공식 (호재 비율 - 악재 비율)
    if total > 0:
        sentiment_score = (pos_score - neg_score) / total
    else:
        sentiment_score = 0.0
        
    return round(sentiment_score, 3), matched_words

# --- 3. 실시간 가상 경제 뉴스 피드 (백테스트 및 실전 시뮬레이션용) ---
NEWS_TEMPLATES = [
    {"headline": "US Tech Giants Beat Earnings Estimates with Surprising Revenue Growth", "impact": "Positive"},
    {"headline": "Inflation Surges Faster Than Expected, Sparking Fears of Tariff Upgrades", "impact": "Negative"},
    {"headline": "Federal Reserve Hints at Imminent Rate Cuts Amid Stable Economic Recovery", "impact": "Positive"},
    {"headline": "Global Supply Chain Slump Causes Huge Profit Loss for Automakers", "impact": "Negative"},
    {"headline": "Major Semiconductor Firm Unveils Revolutionary Breakthrough in AI Chips", "impact": "Positive"},
    {"headline": "Tech Sector Faces Severe Regulatory Investigation and Potential Lawsuit Fine", "impact": "Negative"}
]

# --- 4. 대시보드 레이아웃 구성 ---
st.subheader("📰 실시간 NLP 감성 분석 터미널")

# 사용자 정의 뉴스 입력기
custom_news = st.text_input("📝 직접 테스트할 뉴스 헤드라인을 입력해 보십시오 (영어만 지원):", 
                             value="US Semiconductor sector rebounds strongly after recording profit growth.")

if custom_news:
    score, matches = analyze_sentiment(custom_news)
    col1, col2 = st.columns([1, 3])
    with col1:
        if score > 0.1:
            st.success(f"🟢 호재 감지 (Score: {score})")
        elif score < -0.1:
            st.error(f"🔴 악재 감지 (Score: {score})")
        else:
            st.warning(f"🟡 중립 판독 (Score: {score})")
    with col2:
        st.write(f"🔍 **검출된 금융 핵심 토큰:** {', '.join(matches) if matches else '없음'}")

st.divider()

# --- 5. 백테스팅 시뮬레이션 설정 ---
st.subheader("📊 이벤트 드리븐 감성 매매 시뮬레이터 (S&P 500)")

col_p1, col_p2, col_p3 = st.columns(3)
with col_p1:
    threshold = st.slider("진입 감성 기준치 (Sentiment Threshold)", 0.1, 0.9, 0.3, step=0.05)
with col_p2:
    decay_rate = st.slider("신호 감쇄 강도 (Decay Rate - 일 기준)", 1, 10, 3)
with col_p3:
    leverage = st.selectbox("레버리지 배수 (Leverage)", [1.0, 1.5, 2.0])

# 가상의 30일 시계열 데이터 생성
np.random.seed(42)
dates = pd.date_range(end=datetime.now(), periods=30)
base_price = 5000.0
prices = []
for i in range(30):
    base_price *= (1 + np.random.normal(0.001, 0.01))
    prices.append(base_price)

sim_df = pd.DataFrame({"Date": dates, "Close": prices})
sim_df["Return"] = sim_df["Close"].pct_change().fillna(0)

# 뉴스 이벤트 강제 주입
sim_df["News"] = ""
sim_df["Sentiment"] = 0.0

event_days = [5, 12, 18, 25]
for idx, day in enumerate(event_days):
    template = NEWS_TEMPLATES[idx % len(NEWS_TEMPLATES)]
    sim_df.loc[sim_df.index[day], "News"] = template["headline"]
    score, _ = analyze_sentiment(template["headline"])
    sim_df.loc[sim_df.index[day], "Sentiment"] = score

# --- 감성 매매 로직 수행 ---
positions = []
portfolio_values = [10000.0]  # 시작 자산 10,000 USD
current_position = 0.0        # 0: Cash, 1: Long, -1: Short
decayed_sentiment = 0.0

for i in range(len(sim_df)):
    today_news = sim_df["News"].iloc[i]
    today_sentiment = sim_df["Sentiment"].iloc[i]
    
    # 새로운 뉴스 발생 시 감성 점수 갱신
    if today_news != "":
        decayed_sentiment = today_sentiment
    else:
        # 시간의 흐름에 따른 감성 가치 소멸 (Exponential Decay)
        decayed_sentiment *= (1 - (1 / decay_rate))
        if abs(decayed_sentiment) < 0.05:
            decayed_sentiment = 0.0
            
    # 포지션 결정 (임계치 초과 여부 확인)
    if decayed_sentiment > threshold:
        current_position = 1.0  # 롱 포지션 진입
    elif decayed_sentiment < -threshold:
        current_position = -1.0 # 숏 포지션 진입
    else:
        current_position = 0.0  # 청산 및 현금화
        
    positions.append(current_position)
    
    # 자산 가치 연산 (수익률 반영)
    if i > 0:
        daily_ret = sim_df["Return"].iloc[i]
        prev_value = portfolio_values[-1]
        today_value = prev_value * (1 + (daily_ret * current_position * leverage))
        portfolio_values.append(today_value)

sim_df["Position"] = positions
sim_df["Strategy_Value"] = portfolio_values
sim_df["BuyHold_Value"] = (sim_df["Close"] / sim_df["Close"].iloc[0]) * 10000.0

# 결과 출력
col_res1, col_res2 = st.columns(2)
with col_res1:
    final_roi = (portfolio_values[-1] / 10000.0 - 1) * 100
    bh_roi = (sim_df["BuyHold_Value"].iloc[-1] / 10000.0 - 1) * 100
    st.metric("🏆 NLP 감성 전략 수익률", f"{final_roi:.2f}%", f"단순 보유 대비 {final_roi - bh_roi:+.2f}%")
with col_res2:
    st.metric("📈 S&P 500 단순 보유 수익률", f"{bh_roi:.2f}%")

st.line_chart(sim_df.set_index("Date")[["Strategy_Value", "BuyHold_Value"]])

# 발생 뉴스 타임라인 출력
st.subheader("🕒 감성 뉴스 매매 로그 (Event Log)")
for i in range(len(sim_df)):
    if sim_df["News"].iloc[i] != "":
        pos_str = "🟢 BUY(LONG)" if sim_df["Position"].iloc[i] > 0 else "🔴 SELL(SHORT)"
        st.write(f"📅 **{sim_df['Date'].iloc[i].strftime('%m-%d')}** | {sim_df['News'].iloc[i]} | (Score: {sim_df['Sentiment'].iloc[i]:+.2f}) -> **{pos_str} 실행**")
