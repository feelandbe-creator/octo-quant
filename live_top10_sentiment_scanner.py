import streamlit as st
import pandas as pd
import yfinance as yf
import re
from datetime import datetime

# Streamlit 클라우드 서버의 불필요한 보안 경고문 차단
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 페이지 설정
st.set_page_config(page_title="Top 10 Live Sentiment Scanner", page_icon="🌐", layout="wide")
st.title("🌐 Global Top 10 Live News Sentiment Scanner")
st.caption(f"실시간 뉴스 수집 및 투심 판독 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (KST)")
st.write("미국 및 한국 시가총액 상위 10개 종목의 최신 글로벌 뉴스를 AI가 실시간으로 수집하고 투심을 점수화합니다.")

# --- 1. 글로벌 한/영 통합 금융 특화 감성 사전 ---
EN_LEXICON = {
    "beat": 1.0, "surprise": 0.8, "growth": 0.7, "surge": 0.9, "profit": 0.8,
    "recovery": 0.6, "rebound": 0.6, "upgrade": 0.8, "bullish": 0.9, "record": 0.8,
    "buy": 0.6, "soar": 0.9, "jump": 0.7, "outperform": 0.8, "breakthrough": 0.9,
    "miss": -1.0, "decline": -0.7, "slump": -0.9, "loss": -0.8, "drop": -0.6,
    "downgrade": -0.8, "bearish": -0.9, "deficit": -0.8, "bankruptcy": -1.0,
    "lawsuit": -0.5, "fine": -0.4, "investigation": -0.6, "tariff": -0.7, "inflation": -0.5, "sell": -0.6
}

KR_LEXICON = {
    "어닝 서프라이즈": 1.0, "급등": 0.9, "흑자": 0.9, "상향": 0.8, "돌파": 0.8, 
    "수주": 0.8, "최대 실적": 1.0, "호조": 0.7, "상승": 0.6, "반등": 0.6,
    "회복": 0.6, "배당": 0.6, "인수": 0.5, "혁신": 0.8, "수혜": 0.7, "승인": 0.7,
    "어닝 쇼크": -1.0, "파산": -1.0, "상장폐지": -1.0, "급락": -0.9, "적자": -0.9,
    "하향": -0.8, "악재": -0.8, "침체": -0.8, "부진": -0.7, "관세": -0.7,
    "하락": -0.6, "조사": -0.6, "소송": -0.5, "과징금": -0.5, "인플레이션": -0.5, "매각": -0.4
}

# --- 2. 한/영 하이브리드 NLP 분석 엔진 ---
def analyze_sentiment(text):
    text_lower = text.lower()
    pos_score, neg_score = 0, 0
    matched_words = []
    
    # 영어 분석
    en_words = re.findall(r'\b[a-z]+\b', text_lower)
    for word in en_words:
        if word in EN_LEXICON:
            val = EN_LEXICON[word]
            matched_words.append(f"{word}({val})")
            if val > 0: pos_score += val
            else: neg_score += abs(val)

    # 한국어 분석
    for kr_word, val in KR_LEXICON.items():
        if kr_word in text:
            count = text.count(kr_word)
            for _ in range(count):
                matched_words.append(f"{kr_word}({val})")
                if val > 0: pos_score += val
                else: neg_score += abs(val)
                
    total = pos_score + neg_score
    sentiment_score = (pos_score - neg_score) / total if total > 0 else 0.0
    return round(sentiment_score, 3), matched_words

# --- 3. 타겟 유니버스 (미국 & 한국 Top 10) ---
US_TOP_10 = {
    "Apple": "AAPL", "Microsoft": "MSFT", "NVIDIA": "NVDA", "Alphabet": "GOOGL", 
    "Amazon": "AMZN", "Meta": "META", "Berkshire": "BRK-B", "Tesla": "TSLA", 
    "Eli Lilly": "LLY", "Broadcom": "AVGO"
}

# KOSPI 종목은 야후 파이낸스에서 .KS를 붙여야 검색 가능합니다.
KR_TOP_10 = {
    "삼성전자": "005930.KS", "SK하이닉스": "000660.KS", "LG엔솔": "373220.KS", 
    "삼성바이오": "207940.KS", "현대차": "005380.KS", "기아": "000270.KS", 
    "셀트리온": "068270.KS", "POSCO홀딩스": "005490.KS", "KB금융": "105560.KS", "NAVER": "035420.KS"
}

# --- 4. 자동 뉴스 스크래핑 및 분석 함수 (캐시 적용하여 속도 향상) ---
@st.cache_data(ttl=300) # 5분마다 갱신 (서버 차단 방지)
def scan_news_sentiment(universe_dict):
    results = []
    
    for name, ticker in universe_dict.items():
        try:
            tk = yf.Ticker(ticker)
            news_items = tk.news[:3] # 가장 최신 뉴스 3개만 추출
            
            if not news_items:
                results.append({"종목명": name, "티커": ticker, "투심 점수": 0.0, "상태": "⚪ 중립 (뉴스 없음)", "최신 뉴스 요약": "최근 수집된 뉴스가 없습니다."})
                continue
                
            total_score = 0
            news_details = []
            
            for item in news_items:
                title = item.get('title', '')
                score, tokens = analyze_sentiment(title)
                total_score += score
                # 토큰이 발견되면 뉴스 제목 옆에 표시
                token_str = f" 🔥[{', '.join(tokens)}]" if tokens else ""
                news_details.append(f"[{score:+.2f}] {title}{token_str}")
            
            avg_score = total_score / len(news_items)
            
            # 직관적인 상태 판독
            if avg_score >= 0.3: status = "🟢 강력 매수 (호재)"
            elif avg_score >= 0.1: status = "🟡 긍정적 (호조)"
            elif avg_score <= -0.3: status = "🔴 강력 매도 (악재)"
            elif avg_score <= -0.1: status = "🟠 부정적 (침체)"
            else: status = "⚪ 중립 (특이사항 없음)"

            results.append({
                "종목명": name,
                "티커": ticker,
                "투심 점수": avg_score,
                "상태": status,
                "최신 뉴스 요약": "\n".join(news_details) # 줄바꿈으로 합치기
            })
        except Exception:
            pass # 통신 에러 발생 시 해당 종목 패스
            
    return pd.DataFrame(results)

# --- 5. UI 화면 출력 ---
tab1, tab2 = st.tabs(["🇺🇸 Wall Street Top 10", "🇰🇷 KOSPI Top 10"])

with tab1:
    st.subheader("🇺🇸 미국 시장 시총 상위 10종목 뉴스 투심")
    with st.spinner("미국 글로벌 뉴스망(Yahoo/Bloomberg/Reuters)을 스캔 중입니다..."):
        df_us = scan_news_sentiment(US_TOP_10)
        # 데이터프레임 스타일링 (점수 색상 반영 등)
        st.dataframe(df_us, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("🇰🇷 한국 시장 시총 상위 10종목 뉴스 투심")
    st.info("💡 팁: 한국 종목이라도 글로벌 외신(로이터, 블룸버그)에서 다루는 영문 헤드라인이 수집됩니다. 이중언어 사전이 이를 완벽히 판독합니다.")
    with st.spinner("한국 대표 종목 관련 글로벌 속보를 스캔 중입니다..."):
        df_kr = scan_news_sentiment(KR_TOP_10)
        st.dataframe(df_kr, use_container_width=True, hide_index=True)

st.divider()
st.markdown("**사용 가이드:** 장 시작 전 이 스캐너를 켜서 🟢강력 매수 상태인 종목(호재 발생)과 🔴강력 매도 상태인 종목(돌발 악재)을 1초 만에 파악하고, 오늘의 주도주와 피해야 할 종목을 선별하십시오.")
