import streamlit as st
import pandas as pd
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import re

# --- 1. 금융 특화 NLP 감성 사전 (신규 엔진 동기화) ---
SENTIMENT_DICT = {
    "positive": [
        "급등", "돌파", "상회", "서프라이즈", "흑자", "성장", "최대", "매수", "목표가 상향", "수주", "승인", "호조", "상승", "강세", "기대",
        "surge", "jump", "beat", "exceed", "upgrade", "buy", "growth", "record", "profit", "rally", "outperform", "soar", "approval"
    ],
    "negative": [
        "급락", "하회", "쇼크", "적자", "감소", "최저", "매도", "목표가 하향", "취소", "거절", "부진", "하락", "약세", "우려", "경고", "파산",
        "plunge", "drop", "miss", "downgrade", "sell", "decline", "loss", "warning", "crash", "bankrupt", "underperform", "lawsuit"
    ],
    "magnifiers": [
        "역대", "사상", "폭발적", "초유의", "강력한", "대규모",
        "record", "massive", "strong", "huge", "unprecedented"
    ]
}

# --- 2. Google News RSS 실시간 파싱 엔진 (Top 10) ---
def fetch_top10_news(query):
    search_query = urllib.parse.quote(f"{query} stock OR {query} 주식 OR {query} 실적")
    url = f"https://news.google.com/rss/search?q={search_query}&hl=ko&gl=KR&ceid=KR:ko"
    
    articles = []
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req, timeout=5)
        xml_data = response.read()
        root = ET.fromstring(xml_data)
        
        # 상위 10개 기사 추출
        for item in root.findall('.//item')[:10]:
            title = item.find('title').text
            link = item.find('link').text
            pub_date = item.find('pubDate').text
            articles.append({'title': title, 'link': link, 'date': pub_date})
    except Exception as e:
        st.error(f"뉴스 데이터를 가져오는 중 오류가 발생했습니다: {e}")
    
    return articles

# --- 3. 0~100점 정규화 NLP 분석 알고리즘 ---
def analyze_sentiment_0_to_100(text):
    text_lower = text.lower()
    pos_score = 0
    neg_score = 0
    magnifier_multiplier = 1.0
    
    for word in SENTIMENT_DICT['magnifiers']:
        if word in text_lower:
            magnifier_multiplier = 1.5
            break
            
    for word in SENTIMENT_DICT['positive']:
        if re.search(r'\b' + re.escape(word) + r'\b', text_lower) or word in text_lower:
            pos_score += 1
            
    for word in SENTIMENT_DICT['negative']:
        if re.search(r'\b' + re.escape(word) + r'\b', text_lower) or word in text_lower:
            neg_score += 1
            
    net_score = (pos_score - neg_score) * 15 * magnifier_multiplier
    final_score = max(0, min(100, 50 + net_score)) 
    
    return final_score

# --- 4. UI/UX 렌더링 ---
st.set_page_config(page_title="Top 10 Sentiment Scanner", layout="wide")

st.markdown("""
<style>
    .report-title { font-size: 26px; font-weight: 800; color: #F9FAFB; margin-bottom: 0px; }
    .report-subtitle { font-size: 14px; color: #9CA3AF; margin-bottom: 30px; border-bottom: 2px solid #374151; padding-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="report-title">📰 Live Top 10 Sentiment Scanner</div>', unsafe_allow_html=True)
st.markdown('<div class="report-subtitle">구글 뉴스 상위 10개 헤드라인 실시간 스크래핑 및 0~100 스케일 투심 정량화</div>', unsafe_allow_html=True)

with st.container():
    col1, col2 = st.columns([3, 1])
    with col1:
        target_query = st.text_input("분석할 종목명 또는 티커를 입력하세요", value="NVDA")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🔍 Top 10 뉴스 스캔", use_container_width=True, type="primary")

if analyze_btn and target_query:
    with st.spinner(f"'{target_query}' 관련 Top 10 뉴스 크롤링 및 정규화 분석 중..."):
        
        articles = fetch_top10_news(target_query)
        
        if not articles:
            st.warning("관련 뉴스를 찾을 수 없습니다.")
            st.stop()
            
        total_score = 0
        table_data = []
        
        for article in articles:
            # 0~100 스케일 엔진 적용
            score = analyze_sentiment_0_to_100(article['title'])
            total_score += score
            
            table_data.append({
                '발행일': article['date'],
                '기사 제목': article['title'],
                '투심 점수 (0~100)': score,
                '원문 링크': article['link']
            })
            
        avg_sentiment_score = total_score / len(articles)
        df_results = pd.DataFrame(table_data)
        
        st.markdown("---")
        
        # 종합 점수 대시보드
        score_color = "#EF4444" if avg_sentiment_score >= 60 else ("#3B82F6" if avg_sentiment_score <= 40 else "#9CA3AF")
        sentiment_label = "초강세 (Bullish)" if avg_sentiment_score >= 70 else \
                          "강세 (Positive)" if avg_sentiment_score >= 55 else \
                          "약세 (Negative)" if avg_sentiment_score <= 45 else \
                          "극단적 약세 (Bearish)" if avg_sentiment_score <= 30 else "중립 (Neutral)"
        
        st.markdown(f"""
        <div style="background-color: rgba(255,255,255,0.03); padding: 25px; border-radius: 8px; border: 1px solid #374151; text-align: center; margin-bottom: 30px;">
            <div style="color: #9CA3AF; font-size: 16px; margin-bottom: 10px;">Top 10 기사 종합 NLP 투심 점수 (Mean)</div>
            <div style="color: {score_color}; font-size: 42px; font-weight: 800;">{avg_sentiment_score:.1f}점</div>
            <div style="color: {score_color}; font-size: 18px; margin-top: 5px; font-weight: 700;">{sentiment_label}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 📊 개별 뉴스 투심 데이터표 (Top 10)")
        
        # 데이터프레임 색상 포맷팅 함수 (빨강/파랑/회색)
        def color_sentiment(val):
            if val >= 55: return 'color: #EF4444; font-weight: bold;'
            elif val <= 45: return 'color: #3B82F6; font-weight: bold;'
            else: return 'color: #9CA3AF;'

        # 표 렌더링
        st.dataframe(
            df_results.style.map(color_sentiment, subset=['투심 점수 (0~100)']),
            use_container_width=True,
            column_config={
                "원문 링크": st.column_config.LinkColumn("원문 링크", display_text="기사 보기")
            },
            hide_index=True
        )
