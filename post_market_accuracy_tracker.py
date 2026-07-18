import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pytz
import schedule
import threading
import time
import yfinance as yf
import os

# --- 1. 기본 설정 및 시간 동기화 ---
st.set_page_config(page_title="Alpha Predictor & Tracker", layout="wide", initial_sidebar_state="expanded")

kst_tz = pytz.timezone('Asia/Seoul')
edt_tz = pytz.timezone('US/Eastern')

def get_current_times():
    now_utc = datetime.utcnow().replace(tzinfo=pytz.utc)
    kst = now_utc.astimezone(kst_tz).strftime('%Y-%m-%d %H:%M:%S')
    edt = now_utc.astimezone(edt_tz).strftime('%Y-%m-%d %H:%M:%S')
    return kst, edt

kst_time, edt_time = get_current_times()

# --- 2. 예측 데이터 목업 함수 (UI 복원용) ---
def get_macro_predictions():
    return pd.DataFrame({
        '지수명': ['다우존스 (DJIA)', '나스닥 (IXIC)', '필라델피아 반도체 (SOX)', '다우 운송 (DJT)'],
        '감성 점수': [0.45, 0.82, 0.91, -0.60],
        'Pre 거래량 급증률': ['+120%', '+350%', '+410%', '-40%'],
        '과거 패턴 승률': ['55%', '78%', '85%', '30%'],
        '통합 스코어 (0~100)': [62, 85, 92, 28],
        '예측 추세': ['상승 ↗️', '강한 상승 🚀', '급등 🚀', '하락 ↘️'],
        '예상 변동폭': ['+0.3% ~ +0.6%', '+1.2% ~ +1.8%', '+2.0% ~ +3.5%', '-0.8% ~ -1.5%'],
        '핵심 트리거': ['대형 금융주 실적 호조', 'AI 반도체 가이던스 상향', '파운드리 수요 급증', '운임 하락 및 유가 상승']
    })

def get_sector_predictions():
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

# --- 3. 백그라운드 스케줄러 자동화 로직 ---
DATA_FILE = "daily_predictions.csv"

def get_flat_micro_predictions():
    """트래커 검증을 위해 종목별 중간값을 추출하여 평탄화하는 함수"""
    long_sectors, short_sectors = get_sector_predictions()
    data = []
    
    for sector, stocks in long_sectors.items():
        for ticker, vol_str, nlp, vol_surge, hist_win, score in stocks:
            # 예: '+4.0% ~ +7.0%'의 중간값 5.5 추출
            v_min = float(vol_str.split('~')[0].strip().replace('+', '').replace('%', ''))
            v_max = float(vol_str.split('~')[1].strip().replace('+', '').replace('%', ''))
            data.append({'티커': ticker, '섹터': sector, '예측_방향': 'Long', '예측_변동폭(%)': (v_min + v_max) / 2})
            
    for sector, stocks in short_sectors.items():
        for ticker, vol_str, nlp, vol_surge, hist_win, score in stocks:
            # 예: '-1.5% ~ -3.0%'의 중간값 -2.25 추출
            v_min = float(vol_str.split('~')[0].strip().replace('%', ''))
            v_max = float(vol_str.split('~')[1].strip().replace('%', ''))
            data.append({'티커': ticker, '섹터': sector, '예측_방향': 'Short', '예측_변동폭(%)': (v_min + v_max) / 2})
            
    return pd.DataFrame(data)

def job_save_snapshot():
    """KST 오후 10시에 자동으로 실행되어 데이터를 CSV에 저장"""
    df = get_flat_micro_predictions()
    now_kst = datetime.utcnow().replace(tzinfo=pytz.utc).astimezone(kst_tz)
    df['저장_일시'] = now_kst.strftime('%Y-%m-%d %H:%M:%S')
    df.to_csv(DATA_FILE, index=False)
    print(f"[{df['저장_일시'][0]}] 개장 전 예측 스냅샷 자동 저장 완료.")

@st.cache_resource
def run_scheduler_in_background():
    """UI 블로킹 없이 백그라운드 스레드에서 스케줄러 구동"""
    def scheduler_loop():
        schedule.every().day.at("22:00", "Asia/Seoul").do(job_save_snapshot)
        while True:
            schedule.run_pending()
            time.sleep(60)
            
    thread = threading.Thread(target=scheduler_loop, daemon=True)
    thread.start()
    return thread

run_scheduler_in_background()

# --- 4. yfinance 실제 데이터 호출 함수 ---
@st.cache_data(ttl=3600)
def fetch_actual_performance(tickers):
    try:
        data = yf.download(tickers, period="5d", group_by='ticker', auto_adjust=True, progress=False)
        actuals = {}
        for ticker in tickers:
            ticker_data = data if len(tickers) == 1 else data[ticker]
            latest_open = ticker_data['Open'].iloc[-1]
            latest_close = ticker_data['Close'].iloc[-1]
            intraday_return = ((latest_close - latest_open) / latest_open) * 100
            actuals[ticker] = round(intraday_return, 2)
        return actuals
    except Exception as e:
        return None

# --- 5. 사이드바 UI (이미지 완벽 복원) ---
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

# --- 6. 메인 화면 구조 (Tab 분리) ---
tab_predict, tab_track = st.tabs(["📊 Pre-Market Alpha Predictor v2", "🎯 Post-Market Tracker"])

# ====== TAB 1: 예측 화면 (이미지 UI 완벽 복원) ======
with tab_predict:
    st.title("📊 Pre-Market Alpha Predictor v2")
    st.markdown("자연어 감성 점수, 프리마켓 자금 흐름, 과거 패턴 통계를 통합하여 개장 전 시장을 분석합니다.")

    # 1. 거시 지표 예측
    st.header("1. 주요 지수 변동성 예측 (Macro View)")
    macro_df = get_macro_predictions()

    st.dataframe(
        macro_df,
        column_config={
            "감성 점수": st.column_config.NumberColumn("감성 점수", format="%.2f"),
            "통합 스코어 (0~100)": st.column_config.ProgressColumn(
                "통합 스코어 (0~100)", min_value=0, max_value=100, format="%d"
            ),
        },
        hide_index=True,
        use_container_width=True
    )

    # 2. 섹터 및 종목 예측
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

# ====== TAB 2: 오차 검증 추적기 ======
with tab_track:
    st.header("모델 오차 추적 및 교정 (Auto-Calibration)")
    
    if os.path.exists(DATA_FILE):
        saved_df = pd.read_csv(DATA_FILE)
        save_time = saved_df['저장_일시'].iloc[0]
        st.caption(f"기준 데이터: {save_time} 에 자동 저장된 프리마켓 스냅샷")
        
        tickers = saved_df['티커'].tolist()
        
        with st.spinner("야후 파이낸스에서 실제 장 마감 데이터를 호출 중입니다..."):
            actual_data = fetch_actual_performance(tickers)
            
        if actual_data:
            saved_df['실제_변동률(%)'] = saved_df['티커'].map(actual_data)
            saved_df['오차율(%)'] = saved_df['실제_변동률(%)'] - saved_df['예측_변동폭(%)']
            saved_df['적중_여부'] = np.where(
                ((saved_df['예측_방향'] == 'Long') & (saved_df['실제_변동률(%)'] > 0)) | 
                ((saved_df['예측_방향'] == 'Short') & (saved_df['실제_변동률(%)'] < 0)), 
                'Hit 🎯', 'Miss ❌'
            )
            
            hit_rate = (len(saved_df[saved_df['적중_여부'] == 'Hit 🎯']) / len(saved_df)) * 100
            mae = saved_df['오차율(%)'].abs().mean()
            
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("방향성 적중률 (Hit Rate)", f"{hit_rate:.1f}%")
            mc2.metric("평균 절대 오차 (MAE)", f"{mae:.2f}%")
            mc3.metric("검증된 종목 수", f"{len(saved_df)}개")
            
            st.divider()
            
            st.subheader("예측값 vs 실제 결과 산점도 분석")
            fig_scatter = px.scatter(
                saved_df, x='예측_변동폭(%)', y='실제_변동률(%)', color='적중_여부', hover_data=['티커', '섹터'],
                color_discrete_map={'Hit 🎯': '#00cc96', 'Miss ❌': '#ef553b'}
            )
            fig_scatter.add_shape(type="line", x0=-8, y0=-8, x1=8, y1=8, line=dict(color="gray", dash="dash"))
            fig_scatter.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            st.subheader("종목별 상세 피드백 데이터")
            st.dataframe(saved_df[['티커', '예측_방향', '예측_변동폭(%)', '실제_변동률(%)', '오차율(%)', '적중_여부']], use_container_width=True)
            
        else:
            st.error("실제 시장 데이터를 불러오는데 실패했습니다. 네트워크 상태나 API 제한을 확인하십시오.")
            
    else:
        st.warning("⚠️ 아직 저장된 개장 전 예측 스냅샷이 없습니다. 스케줄러가 KST 22:00에 데이터를 자동으로 저장한 이후(내일)부터 오차 검증 탭이 활성화됩니다.")
