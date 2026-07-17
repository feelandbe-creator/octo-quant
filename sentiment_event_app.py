import streamlit as st
import pandas as pd
import numpy as np
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import re
from datetime import datetime

# --- 1. 금융 특화 NLP 감성 사전 (Financial Sentiment Lexicon) ---
# 기관 투자자들이 주목하는 핵심 호재/악재 키워드 및 가중치 매핑
SENTIMENT_DICT = {
    "positive": [
        "급등", "돌파", "상회", "서프라이즈", "흑자", "성장", "최대", "매수", "목표가 상향", "수주", "승인", "호조", "상승", "강세", "기대",
        "surge", "jump", "beat", "exceed", "upgrade", "buy", "growth", "record", "profit", "rally", "outperform", "soar", "approval"
    ],
    "negative": [
        "급락", "하회", "쇼크", "적자", "감소", "최저", "매도", "목표가 하향", "취소", "거절", "부진", "하락", "약세", "우려", "경고", "파산",
        "plunge", "drop", "miss", "downgrade", "sell", "decline", "loss", "warning", "crash", "bankrupt", "underperform", "lawsuit"
    ],
    "magnifiers": [ # 강조어 (긍정/부정을 증폭)
        "역대", "사상", "폭발적", "초유의", "강력한", "대규모",
        "record", "massive", "strong", "huge", "unprecedented"
    ]
}

# --- 2. Google News RSS 실시간 파싱 엔진 ---
def fetch_top_news(query, num_articles=3):
    # '주식' 또는 'stock' 키워드를 결합하여 금융 뉴스 정확도 향상
    search_query = urllib.parse.quote(f"{query} stock OR {query} 주식 OR {query} 실적")
    url = f"https://news.google.com/rss/search?q={search_query}&hl=ko&gl=KR&ceid=KR:ko"
    
    articles = []
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req, timeout=5)
        xml_data = response.read()
        root = ET.fromstring(xml_data)
        
        for item in root.findall('.//item')[:num_articles]:
            title = item.find('title').text
            link = item.find('link').text
            pub_date = item.find('pubDate').text
            articles.append({'title': title, 'link': link, 'date': pub_date})
    except Exception as e:
        st.error(f"뉴스 데이터를 가져오는 중 오류가 발생했습니다: {e}")
    
    return articles

# --- 3. 경량화 NLP 감성 분석 알고리즘 ---
def analyze_sentiment(text):
    text_lower = text.lower()
    pos_score = 0
    neg_score = 0
    magnifier_multiplier = 1.0
    
    # 강조어 스캔
    for word in SENTIMENT_DICT['magnifiers']:
        if word in text_lower:
            magnifier_multiplier = 1.5
            break
            
    # 긍정어 스캔
    for word in SENTIMENT_DICT['positive']:
        # 정규식을 통해 단어 단위 매칭 가중치 부여
        if re.search(r'\b' + re.escape(word) + r'\b', text_lower) or word in text_lower:
            pos_score += 1
            
    # 부정어 스캔
    for word in SENTIMENT_DICT['negative']:
        if re.search(r'\b' + re.escape(word) + r'\b', text_lower) or word in text_lower:
            neg_score += 1
            
    # 최종 스코어 연산 (기본값 50 기준)
    net_score = (pos_score - neg_score) * 15 * magnifier_multiplier
    final_score = max(0, min(100, 50 + net_score)) # 0 ~ 100 사이로 클리핑
    
    return final_score

# --- 4. UI/UX 렌더링 (다크 테마 리포트 스타일) ---
st.set_page_config(page_title="NLP Sentiment Alpha", layout="wide")

st.markdown("""
<style>
    .report-title { font-size: 26px; font-weight: 800; color: #F9FAFB; margin-bottom: 0px; }
    .report-subtitle { font-size: 14px; color: #9CA3AF; margin-bottom: 30px; border-bottom: 2px solid #374151; padding-bottom: 10px; }
    .metric-card { background-color: rgba(255,255,255,0.03); padding: 20px; border-radius: 8px; border: 1px solid #374151; margin-bottom: 15px; }
    .highlight-red { color: #EF4444; font-weight: 700; }
    .highlight-blue { color: #3B82F6; font-weight: 700; }
    .news-title { font-size: 16px; color: #60A5FA; text-decoration: none; font-weight: 600;}
    .news-title:hover { text-decoration: underline; color: #93C5FD; }
    .news-date { font-size: 12px; color: #9CA3AF; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="report-title">📰 Real-Time NLP Sentiment Alpha</div>', unsafe_allow_html=True)
st.markdown('<div class="report-subtitle">실시간 뉴스 크롤링 및 이벤트 드리븐(Event-Driven) 투심 정량화 엔진</div>', unsafe_allow_html=True)

# 검색 입력부
with st.container():
    col1, col2 = st.columns([3, 1])
    with col1:
        target_query = st.text_input("분석할 종목명 또는 티커를 입력하세요 (예: 애플, TSLA, 삼성전자)", value="TSLA")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🔍 실시간 뉴스 NLP 분석", use_container_width=True, type="primary")

if analyze_btn and target_query:
    with st.spinner(f"'{target_query}' 관련 구글 실시간 뉴스 크롤링 및 투심 분석 중..."):
        
        # 1. 뉴스 데이터 수집
        articles = fetch_top_news(target_query, num_articles=3)
        
        if not articles:
            st.warning("관련 뉴스를 찾을 수 없거나 서버 통신에 실패했습니다. 다른 검색어로 시도해 주세요.")
            st.stop()
            
        # 2. NLP 분석 처리
        total_score = 0
        analyzed_articles = []
        
        for article in articles:
            score = analyze_sentiment(article['title'])
            total_score += score
            analyzed_articles.append({
                'title': article['title'],
                'link': article['link'],
                'date': article['date'],
                'score': score
            })
            
        avg_sentiment_score = total_score / len(articles)
        
        # 3. 분석 결과 표시
        st.markdown("---")
        
        # 상단 대시보드 (NLP 종합 점수)
        score_color = "#EF4444" if avg_sentiment_score >= 60 else ("#3B82F6" if avg_sentiment_score <= 40 else "#9CA3AF")
        sentiment_label = "초강세 (Bullish)" if avg_sentiment_score >= 70 else \
                          "강세 (Positive)" if avg_sentiment_score >= 55 else \
                          "약세 (Negative)" if avg_sentiment_score <= 45 else \
                          "극단적 약세 (Bearish)" if avg_sentiment_score <= 30 else "중립 (Neutral)"
                          
        action_plan = "강력한 호재성 이벤트가 감지되었습니다. 롱(Long) 포지션 진입을 적극 검토하십시오." if avg_sentiment_score >= 60 else \
                      "치명적인 악재성 이벤트가 감지되었습니다. 즉각적인 리스크 관리 및 숏(Short) 포지션 검토가 필요합니다." if avg_sentiment_score <= 40 else \
                      "주가를 움직일 만한 뚜렷한 모멘텀 뉴스가 부재합니다. 기술적 분석에 의존하십시오."
        
        st.markdown(f"""
        <div style="display: flex; gap: 20px; margin-bottom: 20px;">
            <div style="flex: 1; text-align: center; background-color: rgba(255,255,255,0.03); padding: 25px; border-radius: 8px; border: 1px solid #374151;">
                <div style="color: #9CA3AF; font-size: 15px; margin-bottom: 5px;">종합 NLP 투심 점수 (0~100)</div>
                <div style="color: {score_color}; font-size: 36px; font-weight: 800;">{avg_sentiment_score:.1f}점</div>
                <div style="color: {score_color}; font-size: 16px; margin-top: 5px; font-weight: 600;">{sentiment_label}</div>
            </div>
            <div style="flex: 2; display: flex; flex-direction: column; justify-content: center; background-color: rgba(255,255,255,0.03); padding: 25px; border-radius: 8px; border: 1px solid #374151;">
                <div style="color: #F9FAFB; font-size: 18px; font-weight: 700; margin-bottom: 10px;">💡 이벤트 드리븐 액션 플랜</div>
                <div style="color: #D1D5DB; font-size: 15px; line-height: 1.6;">{action_plan}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 추출된 뉴스 리스트 및 개별 점수
        st.markdown("<h4 style='color: #F9FAFB; margin-top: 30px;'>📰 AI 분석 헤드라인 (Top 3)</h4>", unsafe_allow_html=True)
        
        for idx, item in enumerate(analyzed_articles):
            ind_color = "#EF4444" if item['score'] >= 55 else ("#3B82F6" if item['score'] <= 45 else "#9CA3AF")
            st.markdown(f"""
            <div class="metric-card">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div style="flex: 8; padding-right: 15px;">
                        <a href="{item['link']}" target="_blank" class="news-title">[{idx+1}] {item['title']}</a>
                        <div class="news-date">발행: {item['date']}</div>
                    </div>
                    <div style="flex: 1; text-align: right; border-left: 1px solid #4B5563; padding-left: 15px;">
                        <div style="font-size: 12px; color: #9CA3AF;">개별 투심 점수</div>
                        <div style="font-size: 20px; font-weight: 700; color: {ind_color};">{item['score']:.0f}점</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
