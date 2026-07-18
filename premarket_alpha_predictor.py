import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import pytz

# --- 1. 기본 설정 및 데이터 목업 (Mockup) ---
st.set_page_config(page_title="Pre-Market Alpha Predictor", layout="wide", initial_sidebar_state="expanded")

# 한국/뉴욕 시간 표시
kst_tz = pytz.timezone('Asia/Seoul')
edt_tz = pytz.timezone('US/Eastern')
now_utc = datetime.utcnow().replace(tzinfo=pytz.utc)
kst_time = now_utc.astimezone(kst_tz).strftime('%Y-%m-%d %H:%M:%S')
edt_time = now_utc.astimezone(edt_tz).strftime('%Y-%m-%d %H:%M:%S')

# 거시 지표 목업 데이터 생성 함수[cite: 1]
def get_macro_predictions():
    return pd.DataFrame({
        '지수명': ['다우존스 (DJIA)', '나스닥 (IXIC)', '필라델피아 반도체 (SOX)', '다우 운송 (DJT)'],
        '뉴스 감성 (NLP)': [+0.45, +0.82, +0.91, -0.60],
        '예측 추세': ['상승 ↗️', '강한 상승 🚀', '급등 🚀', '하락 ↘️'],
        '예상 변동폭': ['+0.3% ~ +0.6%', '+1.2% ~ +1.8%', '+2.0% ~ +3.5%', '-0.8% ~ -1.5%'],
        '핵심 트리거': ['대형 금융주 실적 호조', 'AI 반도체 가이던스 상향', '파운드리 수요 급증', '운임 하락 및 유가 상승']
    })

# 섹터별 예측 종목 데이터 생성 함수[cite: 1]
def get_sector_predictions():
    long_sectors = {
        'AI 인프라 및 전력': [('VRT', '+4.5% ~ +6.5%'), ('CEG', '+3.0% ~ +4.5%'), ('GEV', '+2.5% ~ +4.0%')],
        '자율주행 및 UAM': [('JOBY', '+4.0% ~ +7.0%'), ('TSLA', '+2.0% ~ +3.5%'), ('ACHR', '+3.0% ~ +5.5%')],
        '사이버 보안': [('CRWD', '+3.5% ~ +5.0%'), ('PANW', '+2.0% ~ +3.5%'), ('ZS', '+1.5% ~ +3.0%')]
    }
    short_sectors = {
        '내수 유통 및 소비재': [('DG', '-3.5% ~ -6.0%'), ('TGT', '-2.0% ~ -3.5%'), ('DLTR', '-3.0% ~ -5.0%')],
        '레거시 자동차': [('F', '-2.5% ~ -4.0%'), ('GM', '-1.5% ~ -3.0%'), ('STLA', '-2.0% ~ -3.5%')],
        '화석 연료 에너지': [('XOM', '-1.0% ~ -2.0%'), ('CVX', '-1.5% ~ -2.5%'), ('OXY', '-2.0% ~ -3.5%')]
    }
    return long_sectors, short_sectors

# --- 2. 사이드바 (설정 및 필터) ---[cite: 1]
st.sidebar.header("⚙️ 설정 및 필터")
st.sidebar.markdown(f"**KST:** {kst_time}")
st.sidebar.markdown(f"**EDT:** {edt_time}")
st.sidebar.divider()

risk_level = st.sidebar.select_slider("리스크 허용도", options=['보수적', '중립', '공격적'], value='중립')
target_market = st.sidebar.multiselect("관심 시장", ['S&P 500', 'NASDAQ 100', 'Russell 2000'], default=['NASDAQ 100'])
auto_refresh = st.sidebar.checkbox("실시간 자동 갱신 (1분 단위)", value=True)

st.sidebar.button("분석 모델 재실행 🔄")

# --- 3. 메인 대시보드 화면 구성 ---[cite: 1]
st.title("📊 Pre-Market Alpha Predictor")
st.markdown("미국장 개장 전 뉴스와 데이터를 분석하여 당일 추세를 예측합니다.")

# 3-1. 거시 지수 예측 렌더링
st.header("1. 주요 지수 변동성 예측 (Macro View)")
macro_df = get_macro_predictions()

st.dataframe(
    macro_df,
    column_config={
        "뉴스 감성 (NLP)": st.column_config.ProgressColumn(
            "감성 점수", help="NLP 기반 뉴스 감성 점수 (-1 ~ 1)", format="%f", min_value=-1, max_value=1
        ),
    },
    hide_index=True,
    use_container_width=True
)

st.divider()

# 3-2. 섹터 및 종목 렌더링 (2단 분리)
st.header("2. 오늘의 주목 섹터 및 종목 (Micro View)")
long_sectors, short_sectors = get_sector_predictions()

col1, col2 = st.columns(2)

with col1:
    st.subheader("🔥 상승 예측 섹터 TOP 3 (Long)")
    for sector, stocks in long_sectors.items():
        with st.expander(f"📈 {sector}"):
            for ticker, volatility in stocks:
                st.markdown(f"- **{ticker}**: {volatility}")

with col2:
    st.subheader("🧊 하락 예측 섹터 TOP 3 (Short)")
    for sector, stocks in short_sectors.items():
        with st.expander(f"📉 {sector}"):
            for ticker, volatility in stocks:
                st.markdown(f"- **{ticker}**: {volatility}")

# 3-3. 감성 점수 게이지 차트 (Plotly)
st.divider()
st.header("3. 실시간 시장 감성 게이지")
fig = go.Figure(go.Indicator(
    mode = "gauge+number+delta",
    value = 0.72,
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "전체 시장 긍정 지수 (Market Sentiment)", 'font': {'size': 20}},
    delta = {'reference': 0.5, 'increasing': {'color': "RebeccaPurple"}},
    gauge = {
        'axis': {'range': [None, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'bar': {'color': "green"},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [0, 0.4], 'color': 'lightcoral'},
            {'range': [0.4, 0.6], 'color': 'lightgray'},
            {'range': [0.6, 1.0], 'color': 'lightgreen'}],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': 0.82}
    }
))
fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "#c9d1d9"}, height=400)
st.plotly_chart(fig, use_container_width=True)
