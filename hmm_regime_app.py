import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 페이지 기본 설정
st.set_page_config(page_title="Wall St. HMM Regime Engine", layout="wide")
st.title("🛡️ Wall Street HMM Regime Switching Model (V2 완전체)")
st.caption(f"판독 기준 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (KST)")

# --- 1. 데이터 수집 함수 (SPY, VIX, TNX 병합) ---
@st.cache_data(ttl=3600)
def fetch_macro_data():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 10) # 10년치 데이터
    
    # yfinance 버전 충돌 및 다중 인덱스 에러를 방지하기 위해 각각 호출 후 병합
    spy_raw = yf.download('SPY', start=start_date, end=end_date)
    vix_raw = yf.download('^VIX', start=start_date, end=end_date)
    tnx_raw = yf.download('^TNX', start=start_date, end=end_date)
    
    # 다중 인덱스 방어 로직 적용하여 'Close' 값만 안전하게 추출
    def get_close(df):
        if isinstance(df.columns, pd.MultiIndex):
            return df['Close'].iloc[:, 0]
        return df['Close']

    df = pd.DataFrame()
    df['SPY'] = get_close(spy_raw)
    df['VIX'] = get_close(vix_raw)
    df['TNX'] = get_close(tnx_raw)
    
    # 결측치(휴장일 차이 등) 제거
    df = df.dropna()
    
    # HMM 인공지능이 학습할 3가지 핵심 피처(Feature) 생성
    df['Return'] = df['SPY'].pct_change() # 주가 수익률
    df['VIX_Level'] = df['VIX']           # VIX 절대 수치 (공포감)
    df['TNX_Diff'] = df['TNX'].diff()     # 10년물 국채 금리 변화량 (유동성/긴축)
    
    df = df.dropna()
    return df

# --- 2. HMM 모델 학습 함수 ---
@st.cache_resource
def fit_hmm_model(df):
    # 입력 데이터: 1.수익률, 2.VIX수치, 3.금리변화
    X = df[['Return', 'VIX_Level', 'TNX_Diff']].values
    
    # 은닉 마르코프 모델(HMM) 생성 (3개의 국면으로 분류)
    model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000, random_state=42)
    model.fit(X)
    
    # 각 날짜별 국면 예측
    hidden_states = model.predict(X)
    df['Regime'] = hidden_states
    
    # 인공지능이 임의로 나눈 0, 1, 2 국면을 VIX(공포지수) 평균치 기준으로 재정렬
    # VIX가 가장 낮은 곳 = 평온(상승장), VIX가 가장 높은 곳 = 공포(폭락장)
    mean_vix_by_regime = [df[df['Regime'] == i]['VIX_Level'].mean() for i in range(3)]
    
    # VIX 평균이 낮은 순서대로 국면 번호(0, 1, 2) 정렬
    sorted_regimes = np.argsort(mean_vix_by_regime)
    
    state_map = {
        sorted_regimes[0]: ('🟢 상승/안정 국면 (Safe)', 'green'),
        sorted_regimes[1]: ('🟡 변동성 확대/조정 국면 (Warning)', 'orange'),
        sorted_regimes[2]: ('🔴 공포/폭락 국면 (Danger - 비중 축소)', 'red')
    }
    
    return df, model, state_map

# --- 3. 메인 로직 및 화면 출력 ---
try:
    with st.spinner("AI가 지난 10년간의 주가, VIX, 금리 데이터를 분석 중입니다..."):
        macro_df = fetch_macro_data()
        analyzed_df, hmm_model, state_map = fit_hmm_model(macro_df)
        
    current_state_idx = analyzed_df['Regime'].iloc[-1]
    current_state_info = state_map[current_state_idx]
    
    # 최신 데이터 추출
    last_spy = analyzed_df['SPY'].iloc[-1]
    last_vix = analyzed_df['VIX'].iloc[-1]
    last_tnx = analyzed_df['TNX'].iloc[-1]
    
    st.subheader("📊 현재 거시 경제 (Macro) 투심 판독 결과")
    
    # 상단 지표 카드
    col1, col2, col3 = st.columns(3)
    col1.metric("S&P 500 (SPY)", f"${last_spy:.2f}", f"{analyzed_df['Return'].iloc[-1]*100:.2f}%")
    col2.metric("VIX (공포지수)", f"{last_vix:.2f}", "30 이상시 극도의 공포", delta_color="inverse")
    col3.metric("미 10년물 금리 (TNX)", f"{last_tnx:.2f}%", f"{analyzed_df['TNX_Diff'].iloc[-1]:.3f}%p", delta_color="inverse")
    
    # AI 판독 결과 박스
    st.markdown(f"""
    <div style="padding: 20px; border-radius: 10px; background-color: {current_state_info[1]}; color: white; text-align: center;">
        <h2 style="margin: 0;">현재 AI 판독 국면: {current_state_info[0]}</h2>
        <p style="margin-top: 10px; font-size: 16px;">VIX와 국채 금리의 복합 흐름을 분석한 결과입니다.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()

    # --- 4. 시각화 (Matplotlib 차트) ---
    st.subheader("📈 10년 주가 흐름 및 AI 국면 감지 히스토리")
    
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # SPY 주가 라인 그리기
    ax.plot(analyzed_df.index, analyzed_df['SPY'], color='black', label='SPY Price', linewidth=1)
    
    # AI가 판독한 국면(Regime)별로 배경색 칠하기
    colors = {0: 'green', 1: 'orange', 2: 'red'}
    for i in range(3):
        # 현재 국면에 해당하는 날짜들만 필터링
        regime_dates = analyzed_df[analyzed_df['Regime'] == i].index
        # vspan을 이용하여 배경색 채우기 (scatter 대신 배경색을 칠하여 직관성 극대화)
        ax.scatter(regime_dates, analyzed_df.loc[regime_dates, 'SPY'], 
                   color=[state_map[r][1] for r in analyzed_df.loc[regime_dates, 'Regime']], 
                   s=10, alpha=0.5, label=state_map[i][0])

    ax.set_title("S&P 500 Price & AI Detected Regimes (VIX + TNX + SPY)", fontsize=16)
    ax.set_xlabel("Date")
    ax.set_ylabel("SPY Price (USD)")
    # 중복 라벨 제거
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    st.info("""
    **💡 V2 엔진 관전 포인트 (모의투자용):** 코로나19 폭락, 2022년 금리인상 발 하락장 등 역사적인 위기 순간에 차트의 점선이 🔴 빨간색(공포 국면)으로 
    정확히 칠해졌는지 확인해 보십시오. 이제 VIX와 금리까지 실시간으로 반영하여 위험을 감지합니다.
    """)

except Exception as e:
    st.error(f"데이터 처리 중 에러가 발생했습니다: {str(e)}")
    st.write("yfinance 데이터 로드 지연일 수 있습니다. 우측 상단의 점 3개를 누르고 'Clear cache'를 실행해 주세요.")
