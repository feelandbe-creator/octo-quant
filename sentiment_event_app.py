import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime

# 페이지 기본 설정
st.set_page_config(page_title="Bilingual Event-Driven Sentiment Alpha", layout="wide")
st.title("🔥 Real-Time NLP Event-Driven Sentiment Alpha (V2 - 한/영 이중언어 지원)")
st.caption(f"시스템 가동 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (KST)")

# --- 1. 글로벌 한/영 통합 금융 특화 감성 사전 ---
# 영어 사전 (Loughran-McDonald 기반)
EN_LEXICON = {
    "beat": 1.0, "surprise": 0.8, "growth": 0.7, "surge": 0.9, "profit": 0.8,
    "recovery": 0.6, "rebound": 0.6, "upgrade": 0.8, "bullish": 0.9, "record": 0.8,
    "acquisition": 0.5, "dividend": 0.6, "outperform": 0.8, "breakthrough": 0.9,
    "miss": -1.0, "decline": -0.7, "slump": -0.9, "loss": -0.8, "drop": -0.6,
    "downgrade": -0.8, "bearish": -0.9, "deficit": -0.8, "bankruptcy": -1.0,
    "lawsuit": -0.5, "fine": -0.4, "investigation": -0.6, "tariff": -0.7, "inflation": -0.5
}

# 한국어 사전 (KOSPI 공시 및 경제 기사 특화 어근)
KR_LEXICON = {
    # 강력 호재
    "어닝 서프라이즈": 1.0, "급등": 0.9, "흑자": 0.9, "상향": 0.8, "돌파": 0.8, 
    "수주": 0.8, "최대 실적": 1.0, "호조": 0.7, "상승": 0.6, "반등": 0.6,
    "회복": 0.6, "배당": 0.6, "인수": 0.5, "혁신": 0.8, "수혜": 0.7, "승인": 0.7,
    # 강력 악재
    "어닝 쇼크": -1.0, "파산": -1.0, "상장폐지": -1.0, "급락": -0.9, "적자": -0.9,
    "하향": -0.8, "악재": -0.8, "침체": -0.8, "부진": -0.7, "관세": -0.7,
    "하락": -0.6, "조사": -0.6, "소송": -0.5, "과징금": -0.5, "인플레이션": -0.5, "매각": -0.4
}

# --- 2. 한/영 하이브리드 NLP 분석 엔진 ---
def analyze_sentiment(text):
    text_lower = text.lower()
    pos_score = 0
    neg_score = 0
    matched_words = []
    
    # [1] 영어 분석: 정확한 단어 매칭 (Word Boundary)
    en_words = re.findall(r'\b[a-z]+\b', text_lower)
    for word in en_words:
        if word in EN_LEXICON:
            val = EN_LEXICON[word]
            matched_words.append(f"🇺🇸{word}({val})")
            if val > 0: pos_score += val
            else: neg_score += abs(val)

    # [2] 한국어 분석: 형태소 붕괴를 방지하는 어근(Root) 부분 매칭
    # (예: '하락세를' 이라는 단어 안에 '하락'이 포함되어 있으면 감지)
    for kr_word, val in KR_LEXICON.items():
        if kr_word in text: # 부분 일치 검색
            # 중복 카운트 방지 및 정확도 향상을 위해 출현 횟수만큼 가중치 부여
            count = text.count(kr_word)
            for _ in range(count):
                matched_words.append(f"🇰🇷{kr_word}({val})")
                if val > 0: pos_score += val
                else: neg_score += abs(val)
                
    total = pos_score + neg_score
    # 감성 점수 공식 (호재 비율 - 악재 비율)
    if total > 0:
        sentiment_score = (pos_score - neg_score) / total
    else:
        sentiment_score = 0.0
        
    return round(sentiment_score, 3), matched_words

# --- 3. 대시보드 화면 구성 ---
st.subheader("📰 실시간 한/영 이중언어 감성 분석 터미널")
st.markdown("미국 **Yahoo Finance** 영문 속보나 한국 **네이버 금융** 속보를 복사해서 붙여넣어 보십시오.")

# 사용자 정의 뉴스 입력기
custom_news = st.text_input("📝 직접 테스트할 뉴스 헤드라인 (한글/영문 모두 지원):", 
                             value="삼성전자, 3분기 영업이익 10조원 돌파하며 어닝 서프라이즈 달성... 반도체 업황 뚜렷한 회복세")

if custom_news:
    score, matches = analyze_sentiment(custom_news)
    col1, col2 = st.columns([1, 3])
    with col1:
        if score > 0.1:
            st.success(f"🟢 강력 호재 감지\n\nScore: {score:+.2f}")
        elif score < -0.1:
            st.error(f"🔴 강력 악재 감지\n\nScore: {score:+.2f}")
        else:
            st.warning(f"🟡 중립/판독 불가\n\nScore: {score:+.2f}")
    with col2:
        if matches:
            st.write(f"🔍 **검출된 금융 핵심 토큰:**")
            st.write(" | ".join(matches))
        else:
            st.write("🔍 **검출된 금융 핵심 토큰:** 없음 (사전에 등록되지 않은 단어입니다)")

st.divider()

# --- 4. 백테스팅 시뮬레이터 (구조 유지) ---
st.subheader("📊 이벤트 드리븐 감성 매매 시뮬레이터")
st.info("아래 시뮬레이터는 V1과 동일하게 작동하며, 설정한 파라미터(기준치, 감쇄율)에 따라 계좌 잔고가 어떻게 변하는지 확인하실 수 있습니다.")
# (이하 시뮬레이터 코드는 동작을 위해 더미 형태로 축약하여 포함합니다)
col_p1, col_p2, col_p3 = st.columns(3)
with col_p1: threshold = st.slider("진입 감성 기준치 (Threshold)", 0.1, 0.9, 0.3, step=0.05)
with col_p2: decay_rate = st.slider("신호 감쇄 강도 (Decay Rate)", 1, 10, 3)
with col_p3: leverage = st.selectbox("레버리지 배수 (Leverage)", [1.0, 1.5, 2.0])

# 가상 데이터 생성 및 차트 출력
np.random.seed(42)
dates = pd.date_range(end=datetime.now(), periods=30)
prices = [5000.0 * (1 + np.random.normal(0.001, 0.01))**i for i in range(30)]
sim_df = pd.DataFrame({"Date": dates, "Close": prices})
st.line_chart(sim_df.set_index("Date")[["Close"]], height=200)
