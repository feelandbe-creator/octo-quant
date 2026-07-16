import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from datetime import datetime, timedelta
import urllib3

# SSL 경고 차단
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(page_title="AI Regime Switching", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 0rem; }
    h1 { color: #1E3A8A; font-size: 1.4rem; }
    .stDataFrame { font-size: 0.85rem; }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ AI Hidden Markov Regime Switching Model")
st.caption(f"판독 기준 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (KST)")
st.divider()

# ==========================================
# [엔진 1] 거시 경제 데이터 수집 (S&P 500 기준)
# ==========================================
@st.cache_data(ttl=86400) # 하루 한 번만 캐싱 (일봉 기준이므로)
def fetch_market_data():
    # 10년간의 S&P 500(SPY) ETF 데이터를 가져와 글로벌 시장 상태를 판독합니다.
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 10)
    
    df = yf.download('SPY', start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False)
    
    # yfinance 최신 버전 다중 인덱스 처리 완벽 방어 로직
    if isinstance(df.columns, pd.MultiIndex):
        close_data = df['Close']
        if isinstance(close_data, pd.DataFrame):
            df = close_data.copy()
            df.columns = ['Close']  # 컬럼 이름을 'Close'로 강제 통일
        else:
            df = close_data.to_frame(name='Close')
    else:
        df = df[['Close']]
        
    # AI에게 학습시킬 '특징(Features)' 추출: 1. 일일 수익률, 2. 10일 변동성
    df['Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Return'].rolling(window=10).std()
    df.dropna(inplace=True)
    return df

# ==========================================
# [엔진 2] HMM (은닉 마르코프 모델) AI 학습
# ==========================================
@st.cache_resource
def train_hmm_model(df):
    # 수익률과 변동성 데이터를 HMM에 맞게 배열 변환
    X = np.column_stack([df['Return'].values, df['Volatility'].values])
    
    # 시장 국면(Regime)을 3가지(상승, 횡보, 하락)로 분류하는 AI 모델 생성
    model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000, random_state=42)
    model.fit(X)
    
    # 과거 10년간의 매일매일이 어떤 국면이었는지 판독 (0, 1, 2 중 하나)
    hidden_states = model.predict(X)
    df['State'] = hidden_states
    
    # 0, 1, 2 로 무작위 배정된 상태를 인간이 이해할 수 있게 '수익률 대비 변동성' 기준으로 정렬
    state_stats = df.groupby('State')['Return'].mean() / df.groupby('State')['Volatility'].mean()
    
    # 샤프 지수(수익률/변동성)가 가장 높은 곳이 1급(안전 상승장), 가장 낮은 곳이 3급(위험 폭락장)
    sorted_states = state_stats.sort_values(ascending=False).index
    state_map = {
        sorted_states[0]: "🟢 국면 1: 안정적 상승장 (Risk-On)",
        sorted_states[1]: "🟡 국면 2: 고변동성 횡보장 (Neutral)",
        sorted_states[2]: "🔴 국면 3: 공포/폭락장 (Risk-Off)"
    }
    
    df['Regime_Name'] = df['State'].map(state_map)
    return df, model, state_map

# 데이터 로드 및 AI 판독 실행
with st.spinner('인공지능이 과거 10년간의 시장 빅데이터를 학습하여 현재 국면을 판독 중입니다...'):
    market_df = fetch_market_data()
    analyzed_df, hmm_model, state_map = train_hmm_model(market_df)

# ==========================================
# [대시보드] 현재 시장 상태 및 투자 비중 제안
# ==========================================
current_regime = analyzed_df['Regime_Name'].iloc[-1]
current_close = analyzed_df['Close'].iloc[-1]
current_vol = analyzed_df['Volatility'].iloc[-1] * 100

st.subheader("🤖 AI 현재 시장 국면 판독 결과")

# 국면에 따른 UI 동적 표시 및 자산 배분 비중 결정
if "🟢" in current_regime:
    st.success(f"현재 시장은 **{current_regime}** 입니다. 주식 비중 확대를 권장합니다.")
    stock_weight, cash_weight = 100, 0
elif "🟡" in current_regime:
    st.warning(f"현재 시장은 **{current_regime}** 입니다. 방향성 탐색 구간이므로 방어적 투자가 필요합니다.")
    stock_weight, cash_weight = 50, 50
else:
    st.error(f"현재 시장은 **{current_regime}** 입니다. 시스템 매매를 중단하고 즉시 현금화하십시오.")
    stock_weight, cash_weight = 0, 100

# 주요 지표 메트릭 표시
col1, col2, col3, col4 = st.columns(4)
col1.metric("SPY 현재가", f"${current_close:.2f}")
col2.metric("현재 시장 단기 변동성", f"{current_vol:.2f}%")
col3.metric("권장 주식(KOSPI/SPY) 비중", f"{stock_weight}%")
col4.metric("권장 현금/채권 비중", f"{cash_weight}%")

st.divider()

# ==========================================
# [시각화] 과거 국면 판독 백테스트 차트
# ==========================================
st.subheader("📈 지난 3년간의 AI 국면 판독 기록 (백테스트)")
st.write("AI가 하락장(빨간색)을 얼마나 정확하게 회피하라고 경고했는지 확인하십시오.")

# 최근 3년(약 750 거래일) 데이터만 시각화
plot_df = analyzed_df.tail(750)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(plot_df.index, plot_df['Close'], color='gray', linewidth=1, label='S&P 500 Index')

# 국면별로 배경색 칠하기
colors = {"🟢": "#c4eed0", "🟡": "#f9ab00", "🔴": "#ffcdd2"}

for state_char, color in colors.items():
    # 해당 국면에 속하는 구간의 인덱스를 찾음
    mask = plot_df['Regime_Name'].str.contains(state_char)
    ax.fill_between(plot_df.index, plot_df['Close'].min(), plot_df['Close'].max(), 
                    where=mask, facecolor=color, alpha=0.3, label=f'Regime {state_char}')

ax.set_title("S&P 500 Price & AI HMM Regimes", fontsize=14)
ax.set_ylabel("Price")
ax.grid(alpha=0.2)

# 중복 라벨 제거용
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='upper left')

st.pyplot(fig)

st.info("""
**[HMM 방어 쉴드 작동 원리]**
* KOSPI는 글로벌 거시 경제(S&P 500)의 큰 흐름을 벗어날 수 없습니다. 
* 본 모델은 S&P 500의 일일 수익률과 변동성을 HMM(은닉 마르코프) AI에 투입하여, 현재 시장이 '안전', '횡보', '위험' 중 어디에 속하는지 매일 추적합니다.
* **빨간색(국면 3)** 이 점등되면 우리가 만든 KOSPI 스캐너의 매수 시그널이 아무리 좋게 나오더라도 **모든 매매를 강제 중지하고 현금을 관망**하는 용도로 사용됩니다.
""")
