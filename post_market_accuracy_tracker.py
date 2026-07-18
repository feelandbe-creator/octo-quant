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
st.set_page_config(page_title="Octo Quant Terminal: Alpha & Tracker", layout="wide")

kst_tz = pytz.timezone('Asia/Seoul')
edt_tz = pytz.timezone('US/Eastern')

def get_current_times():
    now_utc = datetime.utcnow().replace(tzinfo=pytz.utc)
    kst = now_utc.astimezone(kst_tz).strftime('%Y-%m-%d %H:%M:%S')
    edt = now_utc.astimezone(edt_tz).strftime('%Y-%m-%d %H:%M:%S')
    return kst, edt

# --- 2. 예측 엔진 로직 (Mockup API) ---
def generate_market_signals():
    """개장 전 예측 데이터를 생성하는 코어 엔진"""
    data = [
        {'티커': 'JOBY', '섹터': 'UAM', 'NLP_점수': 0.92, 'Volume_증가': 520, '과거승률(%)': 79, '예측_방향': 'Long', '예측_변동폭(%)': 5.5},
        {'티커': 'VRT', '섹터': 'AI인프라', 'NLP_점수': 0.88, 'Volume_증가': 450, '과거승률(%)': 82, '예측_방향': 'Long', '예측_변동폭(%)': 5.5},
        {'티커': 'CRWD', '섹터': '보안', 'NLP_점수': 0.81, 'Volume_증가': 340, '과거승률(%)': 76, '예측_방향': 'Long', '예측_변동폭(%)': 4.25},
        {'티커': 'TSLA', '섹터': '자율주행', 'NLP_점수': 0.60, 'Volume_증가': 210, '과거승률(%)': 65, '예측_방향': 'Long', '예측_변동폭(%)': 2.75},
        {'티커': 'DG', '섹터': '유통', 'NLP_점수': -0.85, 'Volume_증가': 400, '과거승률(%)': 80, '예측_방향': 'Short', '예측_변동폭(%)': -4.75},
        {'티커': 'F', '섹터': '자동차', 'NLP_점수': -0.70, 'Volume_증가': 280, '과거승률(%)': 68, '예측_방향': 'Short', '예측_변동폭(%)': -3.25}
    ]
    df = pd.DataFrame(data)
    # 가중치 연산: NLP(50%), Vol(30%), Hist(20%)
    df['종합_점수'] = (abs(df['NLP_점수']) * 50) + (df['Volume_증가']/100 * 3) * 10 + (df['과거승률(%)'] * 0.2)
    df['종합_점수'] = df['종합_점수'].clip(0, 100).astype(int)
    return df

# --- 3. 백그라운드 스케줄러 자동화 로직 ---
DATA_FILE = "daily_predictions.csv"

def job_save_snapshot():
    """KST 오후 10시에 자동으로 실행되어 데이터를 CSV에 저장하는 함수"""
    df = generate_market_signals()
    now_kst = datetime.utcnow().replace(tzinfo=pytz.utc).astimezone(kst_tz)
    df['저장_일시'] = now_kst.strftime('%Y-%m-%d %H:%M:%S')
    df['대상_거래일'] = now_kst.strftime('%Y-%m-%d')
    
    # CSV 저장 (기존 파일이 있으면 덮어쓰거나 추가 가능. 여기서는 당일 최신화 기준으로 덮어쓰기)
    df.to_csv(DATA_FILE, index=False)
    print(f"[{df['저장_일시'][0]}] 개장 30분 전 예측 스냅샷 자동 저장 완료.")

@st.cache_resource
def run_scheduler_in_background():
    """UI를 블록하지 않도록 스케줄러를 별도 스레드에서 구동"""
    def scheduler_loop():
        # KST 기준 오후 10시 (미국 서머타임 기준 개장 30분 전)
        # 서버 환경에 따라 timezone 이슈 방지를 위해 KST 시간을 직접 체크하여 구동
        schedule.every().day.at("22:00", "Asia/Seoul").do(job_save_snapshot)
        while True:
            schedule.run_pending()
            time.sleep(60) # 1분마다 스케줄 확인
            
    thread = threading.Thread(target=scheduler_loop, daemon=True)
    thread.start()
    return thread

# 앱 구동 시 백그라운드 스레드 시작 (cache되어 한 번만 실행됨)
run_scheduler_in_background()

# --- 4. yfinance 실제 데이터 호출 함수 ---
@st.cache_data(ttl=3600) # 1시간 캐싱하여 API 호출 과부하 방지
def fetch_actual_performance(tickers):
    try:
        data = yf.download(tickers, period="5d", group_by='ticker', auto_adjust=True, progress=False)
        actuals = {}
        for ticker in tickers:
            if len(tickers) == 1:
                ticker_data = data
            else:
                ticker_data = data[ticker]
            # 가장 최근 거래일의 시가 대비 종가 수익률 계산
            latest_open = ticker_data['Open'].iloc[-1]
            latest_close = ticker_data['Close'].iloc[-1]
            intraday_return = ((latest_close - latest_open) / latest_open) * 100
            actuals[ticker] = round(intraday_return, 2)
        return actuals
    except Exception as e:
        return None

# --- 5. UI 및 대시보드 렌더링 ---
kst_time, edt_time = get_current_times()

st.title("🦅 Octo Quant Terminal")
st.markdown(f"**KST:** {kst_time} | **EDT:** {edt_time}")
st.divider()

# 두 개의 독립된 워크스페이스(탭) 생성
tab1, tab2 = st.tabs(["📊 Pre-Market Alpha (개장 전 예측)", "🎯 Post-Market Tracker (장 마감 후 검증)"])

# ====== TAB 1: 프리마켓 예측 ======
with tab1:
    st.header("실시간 시장 강도 및 시그널 분석")
    st.info("💡 매일 KST 22:00 (EDT 09:00)에 백그라운드 스케줄러가 아래 데이터를 자동으로 캡처하여 시스템에 영구 저장합니다.")
    
    current_df = generate_market_signals()
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(
            current_df[['티커', '섹터', 'NLP_점수', 'Volume_증가', '과거승률(%)', '예측_방향', '예측_변동폭(%)', '종합_점수']],
            column_config={
                "종합_점수": st.column_config.ProgressColumn("종합 점수", min_value=0, max_value=100, format="%d"),
                "예측_변동폭(%)": st.column_config.NumberColumn("예측(%)", format="%.2f%%")
            },
            hide_index=True,
            use_container_width=True
        )
    with col2:
        # 시장 통합 강도 게이지
        avg_score = current_df['종합_점수'].mean()
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number", value = avg_score,
            title = {'text': "포트폴리오 통합 롱숏 강도", 'font': {'size': 16}},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#58a6ff"},
                'steps': [{'range': [0, 50], 'color': "lightgray"}, {'range': [50, 100], 'color': "#0d1117"}],
            }
        ))
        fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_gauge, use_container_width=True)

# ====== TAB 2: 포스트마켓 성과 검증 ======
with tab2:
    st.header("모델 오차 추적 및 교정 (Auto-Calibration)")
    
    if os.path.exists(DATA_FILE):
        # 저장된 예측 스냅샷 불러오기
        saved_df = pd.read_csv(DATA_FILE)
        save_time = saved_df['저장_일시'].iloc[0]
        st.caption(f"기준 데이터: {save_time} 에 자동 저장된 스냅샷")
        
        tickers = saved_df['티커'].tolist()
        
        with st.spinner("야후 파이낸스(Yahoo Finance)에서 실제 장 마감 데이터를 호출 중입니다..."):
            actual_data = fetch_actual_performance(tickers)
            
        if actual_data:
            # 예측값과 실제값 병합 로직
            saved_df['실제_변동률(%)'] = saved_df['티커'].map(actual_data)
            saved_df['오차율(%)'] = saved_df['실제_변동률(%)'] - saved_df['예측_변동폭(%)']
            saved_df['적중_여부'] = np.where(
                ((saved_df['예측_방향'] == 'Long') & (saved_df['실제_변동률(%)'] > 0)) | 
                ((saved_df['예측_방향'] == 'Short') & (saved_df['실제_변동률(%)'] < 0)), 
                'Hit 🎯', 'Miss ❌'
            )
            
            hit_rate = (len(saved_df[saved_df['적중_여부'] == 'Hit 🎯']) / len(saved_df)) * 100
            mae = saved_df['오차율(%)'].abs().mean()
            
            # 메트릭 출력
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("방향성 적중률 (Hit Rate)", f"{hit_rate:.1f}%")
            mc2.metric("평균 절대 오차 (MAE)", f"{mae:.2f}%")
            mc3.metric("검증된 종목 수", f"{len(saved_df)}개")
            
            st.divider()
            
            # 괴리율 분석 차트
            st.subheader("예측값 vs 실제 결과 산점도 분석")
            fig_scatter = px.scatter(
                saved_df, x='예측_변동폭(%)', y='실제_변동률(%)', color='적중_여부', hover_data=['티커', '섹터'],
                color_discrete_map={'Hit 🎯': '#00cc96', 'Miss ❌': '#ef553b'}
            )
            fig_scatter.add_shape(type="line", x0=-8, y0=-8, x1=8, y1=8, line=dict(color="gray", dash="dash"))
            fig_scatter.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # 상세 데이터 테이블
            st.subheader("종목별 상세 피드백 데이터")
            st.dataframe(saved_df[['티커', '예측_방향', '예측_변동폭(%)', '실제_변동률(%)', '오차율(%)', '적중_여부']], use_container_width=True)
            
        else:
            st.error("실제 시장 데이터를 불러오는데 실패했습니다. 네트워크 상태나 API 제한을 확인하십시오.")
            
    else:
        st.warning("⚠️ 아직 저장된 개장 전 예측 스냅샷이 없습니다. 스케줄러가 KST 22:00에 데이터를 저장한 이후(내일)부터 오차 검증이 가능합니다.")
