import streamlit as st
import pandas as pd
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import re
from datetime import datetime

# Streamlit 클라우드 서버의 불필요한 보안 경고문 차단
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 페이지 설정
st.set_page_config(page_title="Top 10 Live Sentiment Scanner", page_icon="🌐", layout="wide")
st.title("🌐 Global Top 10 Live News Sentiment Scanner (V2)")
st.caption(f"실시간 뉴스 수집 및 투심 판독 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (KST)")
st.write("미국 및 한국 시가총액 상위 10개 종목의 최신 글로벌 뉴스를 **구글 뉴스망(Google News)**에서 실시간 수집하고 투심을 점수화합니다.")

# --- 1. 글로벌 한/영 통합 금융 특화 감성 사전 (100+ 어휘 대폭 확장판) ---
EN_LEXICON = {
    "beat": 1.0, "beats": 1.0, "beating": 1.0, "surprise": 0.8, "surprises": 0.8, "surprising": 0.8,
    "growth": 0.7, "grow": 0.7, "grows": 0.7, "growing": 0.7, "surge": 0.9, "surges": 0.9, "surging": 0.9,
    "profit": 0.8, "profits": 0.8, "profitable": 0.8, "recovery": 0.6, "recovers": 0.6, "recovering": 0.6,
    "rebound": 0.6, "rebounds": 0.6, "rebounding": 0.6, "upgrade": 0.8, "upgrades": 0.8, "upgraded": 0.8,
    "bullish": 0.9, "bull": 0.5, "record": 0.8, "records": 0.8, "buy": 0.6, "buys": 0.6, "buying": 0.6,
    "soar": 0.9, "soars": 0.9, "soaring": 0.9, "jump": 0.7, "jumps": 0.7, "jumping": 0.7,
    "outperform": 0.8, "outperforms": 0.8, "outperforming": 0.8, "breakthrough": 0.9, "breakthroughs": 0.9,
    "strong": 0.7, "strength": 0.7, "high": 0.5, "higher": 0.5, "gain": 0.6, "gains": 0.6, "gaining": 0.6,
    "win": 0.6, "wins": 0.6, "winning": 0.6, "up": 0.4, "boom": 0.9, "booming": 0.9,
    "miss": -1.0, "misses": -1.0, "missing": -1.0, "decline": -0.7, "declines": -0.7, "declining": -0.7,
    "slump": -0.9, "slumps": -0.9, "slumping": -0.9, "loss": -0.8, "losses": -0.8, "lose": -0.8, "loses": -0.8, "losing": -0.8,
    "drop": -0.6, "drops": -0.6, "dropping": -0.6, "downgrade": -0.8, "downgrades": -0.8, "downgraded": -0.8,
    "bearish": -0.9, "bear": -0.5, "deficit": -0.8, "deficits": -0.8, "bankruptcy": -1.0, "bankrupt": -1.0,
    "lawsuit": -0.5, "lawsuits": -0.5, "sue": -0.5, "sues": -0.5, "sued": -0.5, "fine": -0.4, "fines": -0.4, "fined": -0.4,
    "investigation": -0.6, "investigations": -0.6, "investigate": -0.6, "investigates": -0.6, "investigating": -0.6,
    "tariff": -0.7, "tariffs": -0.7, "inflation": -0.5, "inflationary": -0.5, "sell": -0.6, "sells": -0.6, "selling": -0.6,
    "weak": -0.7, "weaker": -0.7, "weakness": -0.7, "weaken": -0.7, "weakens": -0.7, "weakening": -0.7,
    "cut": -0.6, "cuts": -0.6, "cutting": -0.6, "down": -0.4, "low": -0.5, "lower": -0.5,
    "crash": -1.0, "crashes": -1.0, "crashing": -1.0, "plunge": -0.9, "plunges": -0.9, "plunging": -0.9,
    "risk": -0.6, "risks": -0.6, "risky": -0.6, "fear": -0.7, "fears": -0.7, "worry": -0.6, "worries": -0.6, "worrying": -0.6,
    "fail": -0.8, "fails": -0.8, "failing": -0.8, "failure": -0.8, "delay": -0.5, "delays": -0.5, "delayed": -0.5, "delaying": -0.5,
    "disappoint": -0.8, "disappoints": -0.8, "disappointing": -0.8, "warning": -0.7, "warnings": -0.7, "warn": -0.7, "warns": -0.7,
    "negative": -0.5, "plummet": -0.9, "plummets": -0.9, "plummeting": -0.9, "underperform": -0.8, "underperforms": -0.8, "underperforming": -0.8
}

KR_LEXICON = {
    "서프라이즈": 1.0, "급등": 0.9, "흑자": 0.9, "상향": 0.8, "돌파": 0.8, "수주": 0.8, "최대": 0.8, "호조": 0.7,
    "상승": 0.6, "반등": 0.6, "회복": 0.6, "배당": 0.6, "인수": 0.5, "혁신": 0.8, "수혜": 0.7, "승인": 0.7,
    "강세": 0.7, "기대": 0.5, "확대": 0.6, "성장": 0.7, "개선": 0.6, "유망": 0.6, "돌풍": 0.8, "호실적": 0.9,
    "쇼크": -1.0, "파산": -1.0, "상장폐지": -1.0, "급락": -0.9, "적자": -0.9, "하향": -0.8, "악재": -0.8,
    "침체": -0.8, "부진": -0.7, "관세": -0.7, "하락": -0.6, "조사": -0.6, "소송": -0.5, "과징금": -0.5,
    "인플레": -0.5, "매각": -0.4, "약세": -0.7, "우려": -0.6, "축소": -0.6, "감소": -0.6, "둔화": -0.6,
    "위기": -0.8, "리스크": -0.7, "불확실": -0.6, "지연": -0.5, "철수": -0.6, "중단": -0.7, "매도": -0.6,
    "부도": -1.0, "폭락": -1.0, "경고": -0.7, "실망": -0.8, "어닝쇼크": -1.0, "어닝서프라이즈": 1.0
}

# --- 2. 한/영 하이브리드 NLP 분석 엔진 ---
def analyze_sentiment(text):
    text_lower = text.lower()
    pos_score, neg_score = 0, 0
    matched_words = []
    
    # 영어 분석 (단어 단위 추출)
    en_words = re.findall(r'\b[a-z]+\b', text_lower)
    for word in en_words:
        if word in EN_LEXICON:
            val = EN_LEXICON[word]
            matched_words.append(f"{word}({val})")
            if val > 0: pos_score += val
            else: neg_score += abs(val)

    # 한국어 분석 (어근 부분 일치 검색으로 형태소 붕괴 방어)
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

KR_TOP_10 = {
    "삼성전자": "005930.KS", "SK하이닉스": "000660.KS", "LG엔솔": "373220.KS", 
    "삼성바이오": "207940.KS", "현대차": "005380.KS", "기아": "000270.KS", 
    "셀트리온": "068270.KS", "POSCO홀딩스": "005490.KS", "KB금융": "105560.KS", "NAVER": "035420.KS"
}

# --- 4. 자동 뉴스 스크래핑 및 분석 함수 (구글 뉴스 RSS 엔진 탑재) ---
@st.cache_data(ttl=300) # 5분마다 갱신 (서버 차단 방지)
def scan_news_sentiment(universe_dict, is_kr=False):
    results = []
    
    for name, ticker in universe_dict.items():
        try:
            # 야후 API 오류 회피를 위해 빠르고 절대 막히지 않는 구글 뉴스 RSS 활용
            query = f"{name} 주식 뉴스" if is_kr else f"{name} stock news"
            encoded_query = urllib.parse.quote(query)
            # 한국어는 한국 구글, 영어는 미국 구글로 분기 처리
            url = f"https://news.google.com/rss/search?q={encoded_query}&hl={'ko&gl=KR&ceid=KR:ko' if is_kr else 'en-US&gl=US&ceid=US:en'}"
            
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            response = urllib.request.urlopen(req, timeout=5)
            root = ET.fromstring(response.read())
            
            news_items = []
            for item in root.findall('./channel/item')[:3]:
                title = item.find('title').text
                # 구글 뉴스의 끝부분 출처 표기 제거 (예: " - 머니투데이" 또는 " - Yahoo Finance")
                if ' - ' in title:
                    title = title.rsplit(' - ', 1)[0]
                news_items.append(title)
            
            if not news_items:
                results.append({
                    "종목명": name, 
                    "티커": ticker, 
                    "종합 NLP 투심 점수 (0~100)": 50, 
                    "상태": "⚪ 중립 (뉴스 없음)", 
                    "최신 뉴스 요약": "최근 수집된 뉴스가 없습니다."
                })
                continue
                
            total_score = 0
            news_details = []
            
            for title in news_items:
                score, tokens = analyze_sentiment(title)
                total_score += score
                # 토큰이 발견되면 뉴스 제목 옆에 직관적으로 불꽃 마크 표시
                token_str = f" 🔥[{', '.join(tokens)}]" if tokens else ""
                news_details.append(f"[{score:+.2f}] {title}{token_str}")
            
            avg_score = total_score / len(news_items)
            
            # -1.0 ~ +1.0의 점수를 0 ~ 100 점수로 직관적으로 변환 (50점이 중립)
            nlp_score_100 = int(round((avg_score + 1.0) * 50))
            nlp_score_100 = max(0, min(100, nlp_score_100)) # 0~100 사이 보정
            
            # 직관적인 상태 판독
            if avg_score >= 0.3: status = "🟢 강력 매수 (호재)"
            elif avg_score >= 0.1: status = "🟡 긍정적 (호조)"
            elif avg_score <= -0.3: status = "🔴 강력 매도 (악재)"
            elif avg_score <= -0.1: status = "🟠 부정적 (침체)"
            else: status = "⚪ 중립 (특이사항 없음)"

            results.append({
                "종목명": name,
                "티커": ticker,
                "종합 NLP 투심 점수 (0~100)": nlp_score_100,
                "상태": status,
                "최신 뉴스 요약": "\n".join(news_details) # 줄바꿈으로 합치기
            })
        except Exception as e:
            results.append({
                "종목명": name, 
                "티커": ticker, 
                "종합 NLP 투심 점수 (0~100)": 50, 
                "상태": "⚪ 통신 에러", 
                "최신 뉴스 요약": f"뉴스 수집 실패: {str(e)}"
            })
            
    return pd.DataFrame(results)

# --- 5. UI 화면 출력 ---
tab1, tab2 = st.tabs(["🇺🇸 Wall Street Top 10", "🇰🇷 KOSPI Top 10"])

with tab1:
    st.subheader("🇺🇸 미국 시장 시총 상위 10종목 뉴스 투심")
    with st.spinner("미국 글로벌 구글 뉴스망(Google News)을 스캔 중입니다..."):
        df_us = scan_news_sentiment(US_TOP_10, is_kr=False)
        st.dataframe(df_us, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("🇰🇷 한국 시장 시총 상위 10종목 뉴스 투심")
    st.info("💡 팁: 한국 종목은 로컬 구글 뉴스(한글)를 스캔하여 한국어 사전을 통해 판독합니다.")
    with st.spinner("한국 구글 뉴스망을 스캔 중입니다..."):
        df_kr = scan_news_sentiment(KR_TOP_10, is_kr=True)
        st.dataframe(df_kr, use_container_width=True, hide_index=True)

st.divider()
st.markdown("**사용 가이드:** 장 시작 전 이 스캐너를 켜서 🟢강력 매수 상태인 종목(호재 발생)과 🔴강력 매도 상태인 종목(돌발 악재)을 1초 만에 파악하고, 오늘의 주도주와 피해야 할 종목을 선별하십시오.")
