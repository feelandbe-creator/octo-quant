import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import pytz

# --- 1. 기본 설정 및 시간 동기화 ---
st.set_page_config(page_title="Pre-Market Alpha Predictor v2", layout="wide", initial_sidebar_state="expanded")

kst_tz = pytz.timezone('Asia/Seoul')
edt_tz = pytz.timezone('US/Eastern')
now_utc = datetime.utcnow().replace(tzinfo=pytz.utc)
kst_time = now_utc.astimezone(kst_tz).strftime('%Y-%m-%d %H:%M:%S')
edt_time = now_utc.astimezone(edt_tz).strftime('%Y-%m-%d %H:%M:%S')

# --- 2. 다중 변수 통합 데이터 목업 (Volume & Historical 추가) ---
def get_macro_predictions():
    # 종합 예측 점수 = (NLP * 0.5) + (Volume * 0.3) + (Historical * 0.2) 가상 산출값
    return pd.DataFrame({
        '지수명': ['다우존스 (DJIA)', '나스닥 (IXIC)', '필라델피아 반도체 (SOX)', '다우 운송 (DJT)'],
        '뉴스 감성 (NLP)': [+0.45, +0.82, +0.91, -0.60],
        'Pre 거래량 급증률': ['+120%', '+350%', '+410%', '-40%'],
        '과거 패턴 승률': ['55%', '78%', '85%', '30%'],
        '종합 예측 점수': [62, 85, 92, 28], # 100점 만점 기준 환산
        '예측 추세': ['상승 ↗️', '강한 상승 🚀', '급등 🚀', '하락 ↘️'],
        '예상 변동폭': ['+0.3% ~ +0.6%', '+1.2% ~ +1.8%', '+2.0% ~ +3.5%', '-0.8% ~ -1.5%'],
        '핵심 트리거': ['대형 금융주 실적 호조', 'AI 반도체 가이던스 상향', '파운드리 수요 급증', '운임 하락 및 유가 상승']
    })

def get_sector_predictions():
    # 구조: 종목 기호: (예상 변동폭, 뉴스 감성, 프리마켓 거래량 급증률, 과거 패턴 승률, 종합 예측 점수)
    long_sectors = {
        'AI 인프라 및 전력': [
            ('VRT', '+4.5% ~ +6.5%', 0.88, '+450%', '82%', 89), 
            ('CEG', '+3.0% ~ +4.5%', 0.75, '+280%', '71%', 77), 
            ('GEV', '+2.5% ~ +4.0%', 0.65, '+190%', '68%', 72)
        ],
        '자율주행 및 UAM': [
            ('JOBY', '+4.0% ~ +7.0%', 0.92, '+520%', '79%', 91), 
            ('TSLA', '+2.0% ~ +3.5%', 0.60, '+210%', '65%', 68), 
            ('ACHR', '+3.0% ~ +5.5%', 0.78, '+310%', '74%', 81)
        ],
        '사이버 보안': [
            ('CRWD', '+3.5% ~ +5.0%', 0.81, '+340%', '76%', 84), 
            ('PANW', '+2.0% ~ +3.5%', 0.55, '+150%', '62%', 64), 
            ('ZS', '+1.5% ~ +3.0%', 0.50, '+120%', '58%', 60)
        ]
    }
    short_sectors = {
        '내수 유통 및 소비재': [
            ('DG', '-3.5% ~ -6.0%', -0.85, '+400%', '80%', 12), 
            ('TGT', '-2.0% ~ -3.5%', -0.60, '+220%', '65%', 35), 
            ('DLTR', '-3.0% ~ -5.0%', -0.75, '+310%', '72%', 22)
        ],
        '레거시 자동차': [
            ('F', '-2.5% ~ -4.0%', -0.70, '+280%', '68%', 28), 
            ('GM', '-1.5% ~ -3.0%', -0.45, '+150%', '55%', 42), 
            ('STLA', '-2.0% ~ -3.5%', -0.55, '+190%', '60%', 38)
        ]
    }
    return long_sectors, short_sectors

# --- 3. 사이드바 (설정 및 필터) ---
st.sidebar.header("⚙️ 시스템 설정")
st.sidebar.markdown(f"**KST:** {kst_time}")
st.sidebar.markdown(f"**EDT:** {edt_time}")
st.sidebar.divider()

st.sidebar.subheader("가중치 설정 (Weight Tuning)")
weight_nlp = st.sidebar.slider("뉴스 감성 (NLP) 가중치", 0.0, 1.0, 0.5, 0.1)
weight_vol = st.sidebar.slider("프리마켓 거래량 가중치", 0.0, 1.0, 0.3, 0.1)
weight_hist = st.sidebar.slider("과거 패턴 승률 가중치", 0.0, 1.0, 0.2, 0.1)

st.sidebar.info("총합이 1.0이 되도록 조정하여 백테스팅 로직에 반영합니다.")
st.sidebar.button("분석 모델 재실행 🔄")

# --- 4. 메인 대시보드 화면 구성 ---
st.title("📊 Pre-Market Alpha Predictor v2")
st.markdown("자연어 감성 점수, 프리마켓 자금 흐름, 과거 패턴 통계를 통합하여 개장 전 시장을 분석합니다.")

# 4-1. 거시 지수 예측 (Macro View)
st.header("1. 주요 지수 변동성 예측 (Macro View)")
macro_df = get_macro_predictions()

st.dataframe(
    macro_df,
    column_config={
        "뉴스 감성 (NLP)": st.column_config.NumberColumn("감성 점수", format="%.2f"),
        "종합 예측 점수": st.column_config.ProgressColumn(
            "통합 스코어 (0~100)", help="NLP, 거래량, 과거승률을 결합한 매수/매도 강도", format="%d", min_value=0, max_value=100
        ),
    },
    hide_index=True,
    use_container_width=True
)

st.divider()

# 4-2. 섹터 및 종목 렌더링 (Micro View)
st.header("2. 다중 지표 기반 주목 종목 (Micro View)")
long_sectors, short_sectors = get_sector_predictions()

col1, col2 = st.columns(2)

with col1:
    st.subheader("🔥 강력 매수 섹터 TOP (Long)")
    for sector, stocks in long_sectors.items():
        with st.expander(f"📈 {sector}"):
            for ticker, volatility, nlp, vol_surge, hist_win, score in stocks:
                st.markdown(f"**{ticker}** | 예상: {volatility}")
                st.caption(f"↳ 감성: {nlp} | 거래량 증가: {vol_surge} | 과거 승률: {hist_win} ➔ **종합점수: {score}점**")

with col2:
    st.subheader("🧊 강력 매도/관망 섹터 TOP (Short)")
    for sector, stocks in short_sectors.items():
        with st.expander(f"📉 {sector}"):
            for ticker, volatility, nlp, vol_surge, hist_win, score in stocks:
                st.markdown(f"**{ticker}** | 예상: {volatility}")
                st.caption(f"↳ 감성: {nlp} | 거래량 증가: {vol_surge} | 과거 승률: {hist_win} ➔ **종합점수: {score}점**")

# 4-3. 감성 점수 게이지 차트 (Plotly)
st.divider()
st.header("3. 실시간 통합 시장 강도 (Market Momentum)")
fig = go.Figure(go.Indicator(
    mode = "gauge+number+delta",
    value = 82, # 종합 예측 점수 기반 수치로 변경
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "시장 통합 강도 지수 (0~100)", 'font': {'size': 20}},
    delta = {'reference': 50, 'increasing': {'color': "RebeccaPurple"}},
    gauge = {
        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'bar': {'color': "#00cc96"},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [0, 40], 'color': '#ef553b'},
            {'range': [40, 60], 'color': '#e5ecf6'},
            {'range': [60, 100], 'color': '#00cc96'}],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': 85}
    }
))
fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "#c9d1d9"}, height=400)
st.plotly_chart(fig, use_container_width=True)
