import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import datetime

# --- 1. 초정밀 개별 다운로드 및 병합 엔진 ---
@st.cache_data(show_spinner=False)
def fetch_comprehensive_market_data(tickers, start_date, end_date):
    compiled_data = pd.DataFrame()
    
    for ticker in tickers:
        try:
            raw_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if raw_data.empty:
                continue
                
            if 'Adj Close' in raw_data.columns:
                price_series = raw_data['Adj Close']
            elif 'Close' in raw_data.columns:
                price_series = raw_data['Close']
            else:
                continue
                
            compiled_data[ticker] = price_series
        except Exception:
            continue
            
    compiled_data.ffill(inplace=True)
    compiled_data.dropna(inplace=True)
    return compiled_data

# --- 2. 동적 가중치 적용 멀티 매칭 알고리즘 ---
def find_top_historical_matches(df, macro_tickers, target_stock, window_size, top_n=3):
    if target_stock not in df.columns:
        return [], {}
        
    valid_macros = [t for t in macro_tickers if t in df.columns]
    if not valid_macros:
        return [], {}
        
    recent_window_data = df.iloc[-window_size:]
    correlations = {}
    for ticker in valid_macros:
        corr = recent_window_data[target_stock].corr(recent_window_data[ticker])
        correlations[ticker] = abs(corr) if not np.isnan(corr) else 0.0
        
    total_corr = sum(correlations.values())
    if total_corr > 0:
        weights = {k: v / total_corr for k, v in correlations.items()}
    else:
        weights = {k: 1.0 / len(valid_macros) for k in valid_macros}
        
    macro_data = df[valid_macros]
    scaler = StandardScaler()
    scaled_macro = scaler.fit_transform(macro_data)
    
    weight_vector = np.array([np.sqrt(weights[ticker]) for ticker in valid_macros])
    weighted_scaled_macro = scaled_macro * weight_vector
    
    current_pattern = weighted_scaled_macro[-window_size:]
    historical_data = weighted_scaled_macro[:-window_size - 20]
    
    distances = []
    for i in range(len(historical_data) - window_size):
        past_window = historical_data[i : i + window_size]
        distance, _ = fastdtw(current_pattern, past_window, dist=euclidean)
        distances.append((i, distance))
        
    distances.sort(key=lambda x: x[1])
    
    top_matches = []
    selected_indices = []
    for idx, dist in distances:
        if len(top_matches) >= top_n:
            break
        if any(abs(idx - s_idx) < window_size for s_idx in selected_indices):
            continue
        top_matches.append((idx, dist))
        selected_indices.append(idx)
        
    return top_matches, weights

# --- 3. 엔터프라이즈급 UI/UX 대시보드 ---
st.set_page_config(page_title="옥토만경님 전용 - 초정밀 모멘텀 시스템", layout="wide")
st.title("🛡️ 옥토만경님 전용: 다차원 동적 가중치 기반 초정밀 모멘텀 예측 시스템")
st.markdown("정확성 최우선 모드: 7대 글로벌 매크로 자산 벡터와 상관관계 가중치 매칭 알고리즘을 가동합니다.")

st.sidebar.header("🎛️ 알고리즘 제어 핵심 설정")
target_stock = st.sidebar.text_input("분석 대상 종목 티커 (예: NVDA, TSLA, 005930.KS)", value="NVDA").upper().strip()
window = st.sidebar.slider("추세 분석 윈도우 (최근 N일간의 흐름)", min_value=15, max_value=90, value=45)
lookback_years = st.sidebar.slider("역사적 데이터 탐색 깊이 (년)", min_value=5, max_value=20, value=12)

if st.sidebar.button("⚙️ 고정밀 시뮬레이션 개시"):
    with st.spinner('고정밀 매크로 데이터 수집 및 다차원 DTW 행렬 연산을 수행 중입니다. 잠시만 기다려 주십시오...'):
        
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=365 * lookback_years)
        
        macro_tickers = ['QQQ', '^GSPC', 'DIA', '^TNX', 'DX=F', '^VIX', 'CL=F']
        all_tickers = list(set(macro_tickers + [target_stock]))
        
        df = fetch_comprehensive_market_data(all_tickers, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        
        if target_stock not in df.columns:
            st.error(f"⚠️ 야후 파이낸스에서 '{target_stock}' 종목의 데이터를 불러오지 못했습니다. 티커명이 정확한지 확인해 주십시오.")
            st.stop()
            
        top_matches, feature_weights = find_top_historical_matches(df, macro_tickers, target_stock, window_size=window, top_n=3)
        
        if not top_matches:
            st.error("⚠️ 데이터를 연산하기 위한 과거 시계열이 충분하지 않습니다. 과거 탐색 기간을 늘리거나 다른 종목을 선택해 주십시오.")
            st.stop()
        
        st.subheader("🎯 1. 현재 시장 지배 지표 분석 (Dynamic Weights)")
        st.markdown(f"최근 {window}일 동안 **{target_stock}**의 주가 움직임과 가장 연동성이 높았던 매크로 지표별 가중치입니다. 알고리즘이 자동으로 패턴 매칭 시 이 지표들의 형상에 가중치를 부여했습니다.")
        
        weight_fig = go.Figure([go.Bar(
            x=[f"{k} (금리/지수/원자재)" for k in feature_weights.keys()], 
            y=list(feature_weights.values()),
            marker_color='rgb(26, 54, 93)'
        )])
        weight_fig.update_layout(yaxis_title="정규화 가중치 비중", xaxis_tickangle=-15, height=300)
        st.plotly_chart(weight_fig, use_container_width=True)
        
        st.subheader("🔮 2. 고정밀 앙상블 패턴 매칭 및 향후 20일 시나리오")
        st.markdown("과거 데이터 중 현재의 매크로 다차원 흐름과 가장 일치하는 독립적 과거 시점 3곳의 통계적 추적 데이터입니다.")
        
        returns_list = []
        cols = st.columns(3)
        
        for rank, (match_idx, dist_score) in enumerate(top_matches):
            m_start = df.index[match_idx].strftime('%Y-%m-%d')
            m_end = df.index[match_idx + window].strftime('%Y-%m-%d')
            
            p_curr = df[target_stock].iloc[match_idx + window]
            p_future = df[target_stock].iloc[match_idx + window + 20]
            ret = ((p_future - p_curr) / p_curr) * 100
            returns_list.append(ret)
            
            with cols[rank]:
                st.info(f"**[상위 {rank+1}위 유사 매칭 구간]**")
                st.metric("유사도 거리 점수", f"{dist_score:.2f}")
                st.write(f"📅 구간: {m_start} ~ {m_end}")
                st.metric("해당 국면 직후 20일 수익률", f"{ret:.2f}%", delta=f"{ret:.2f}%")
        
        st.subheader("📊 3. 종합 통계적 모멘텀 기대치")
        avg_return = np.mean(returns_list)
        win_rate = sum(1 for r in returns_list if r > 0) / len(returns_list) * 100
        
        c1, c2, c3 = st.columns(3)
        c1.metric("앙상블 평균 예상 수익률 (20일 기대치)", f"{avg_return:.2f}%")
        c2.metric("통계적 상승 확률 (Win Rate)", f"{win_rate:.1f}%")
        c3.metric("최대 업사이드 / 다운사이드", f"{max(returns_list):.1f}% / {min(returns_list):.1f}%")
        
        st.subheader(f"📈 4. {target_stock}의 향후 예상 주가 경로 시뮬레이션")
        path_fig = go.Figure()
        
        curr_series = df[target_stock].iloc[-window:].values
        curr_norm = (curr_series - np.min(curr_series)) / (np.max(curr_series) - np.min(curr_series))
        path_fig.add_trace(go.Scatter(y=curr_norm, mode='lines', name='현재 실제 주가 경로 (최근)', line=dict(color='red', width=4)))
        
        for rank, (match_idx, _) in enumerate(top_matches):
            past_full_series = df[target_stock].iloc[match_idx : match_idx + window + 20].values
            past_norm = (past_full_series - np.min(past_full_series[:window])) / (np.max(past_full_series[:window]) - np.min(past_full_series[:window]))
            
            path_fig.add_trace(go.Scatter(
                y=past_norm, 
                mode='lines', 
                name=f'상위 {rank+1}위 과거 시나리오 경로', 
                line=dict(dash='dash', width=2)
            ))
            
        path_fig.add_shape(type="line", x0=window-1, y0=0, x1=window-1, y1=2, line=dict(color="black", width=2, dash="dot"))
        
        # [수정 완료 구간] Plotly 문법 오류(argv) 수정 및 화살표 정상 출력 
        path_fig.add_annotation(x=window-1, y=1.5, text="현재 분석 시점 (이후는 과거의 미래 경로)", showarrow=True, arrowhead=1)
        
        path_fig.update_layout(title=f"{target_stock}의 국면 정규화 시나리오 프로젝션", xaxis_title="경과 일수 (Days)", yaxis_title="정규화 스케일 가격")
        st.plotly_chart(path_fig, use_container_width=True)
        
        st.success("🔒 본 리포트는 개별 종목 정밀 추적 엔진과 7대 거시 지표의 동적 가중치 연산 결과입니다. 옥토만경님의 정밀 자산 운용을 위한 정량적 기초 자료로 활용하십시오.")
