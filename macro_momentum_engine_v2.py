import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import datetime

# --- 1. 데이터 수집 및 정제 엔진 ---
@st.cache_data(show_spinner=False)
def fetch_comprehensive_market_data(tickers, start_date, end_date):
    """
    Yahoo Finance API를 통해 매크로 및 개별 종목 데이터를 다운로드하고
    결측치 처리 및 동기화를 수행합니다.
    """
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    # 전일 데이터로 결측치 대체 (Forward Fill) 후 남은 결측치 제거
    data.fillna(method='ffill', inplace=True)
    data.dropna(inplace=True)
    return data

# --- 2. 동적 가중치 적용 멀티 매칭 알고리즘 ---
def find_top_historical_matches(df, macro_tickers, target_stock, window_size, top_n=3):
    """
    상관관계 기반 동적 가중치와 FastDTW를 결합하여
    가장 유사한 상위 N개의 과거 국면을 도출합니다 (중복 구간 배제).
    """
    # 최근 윈도우 기간 동안의 상관관계 계산 (가중치 산출용)
    recent_window_data = df.iloc[-window_size:]
    correlations = {}
    for ticker in macro_tickers:
        corr = recent_window_data[target_stock].corr(recent_window_data[ticker])
        correlations[ticker] = abs(corr) if not np.isnan(corr) else 0.0
        
    # 가중치 정규화 (합이 1이 되도록 설정)
    total_corr = sum(correlations.values())
    if total_corr > 0:
        weights = {k: v / total_corr for k, v in correlations.items()}
    else:
        weights = {k: 1.0 / len(macro_tickers) for k in macro_tickers}
        
    # 매크로 데이터 분리 및 정규화 (Z-Score)
    macro_data = df[macro_tickers]
    scaler = StandardScaler()
    scaled_macro = scaler.fit_transform(macro_data)
    
    # 거리에 가중치를 반영하기 위해 데이터에 루트 가중치(sqrt(w))를 사전 곱 연산
    # 수식: w * (x - y)^2 = (sqrt(w)*x - sqrt(w)*y)^2
    weight_vector = np.array([np.sqrt(weights[ticker]) for ticker in macro_tickers])
    weighted_scaled_macro = scaled_macro * weight_vector
    
    # 현재 패턴과 과거 풀 데이터 분리
    current_pattern = weighted_scaled_macro[-window_size:]
    # 미래 20일 수익률 계산을 확보하기 위해 과거 탐색 범위를 제한
    historical_data = weighted_scaled_macro[:-window_size - 20]
    
    distances = []
    # 슬라이딩 윈도우 방식으로 과거 전 구간 탐색
    for i in range(len(historical_data) - window_size):
        past_window = historical_data[i : i + window_size]
        # FastDTW 거리 연산 (거리가 짧을수록 고유사)
        distance, _ = fastdtw(current_pattern, past_window, dist=euclidean)
        distances.append((i, distance))
        
    # 거리 기준 오름차순 정렬
    distances.sort(key=lambda x: x[1])
    
    # 상위 N개 비중복 구간 추출 (윈도우가 겹치지 않도록 필터링)
    top_matches = []
    selected_indices = []
    for idx, dist in distances:
        if len(top_matches) >= top_n:
            break
        # 기존에 선택된 구간과 겹치는지 검증 (윈도우 크기 기준 포개짐 방지)
        if any(abs(idx - s_idx) < window_size for s_idx in selected_indices):
            continue
        top_matches.append((idx, dist))
        selected_indices.append(idx)
        
    return top_matches, weights

# --- 3. 엔터프라이즈급 UI/UX 대시보드 ---
st.set_page_config(page_title="옥토만경님 전용 - 초정밀 모멘텀 시스템", layout="wide")
st.title("🛡️ 옥토만경님 전용: 다차원 동적 가중치 기반 초정밀 모멘텀 예측 시스템")
st.markdown("정확성 최우선 모드: 7대 글로벌 매크로 자산 벡터와 상관관계 가중치 매칭 알고리즘을 가동합니다.")

# 사이드바 제어 패널
st.sidebar.header("🎛️ 알고리즘 제어 핵심 설정")
target_stock = st.sidebar.text_input("분석 대상 종목 티커 (예: NVDA, TSLA, 005930.KS)", value="NVDA").upper()
window = st.sidebar.slider("추세 분석 윈도우 (최근 N일간의 흐름)", min_value=15, max_value=90, value=45)
lookback_years = st.sidebar.slider("역사적 데이터 탐색 깊이 (년)", min_value=5, max_value=20, value=12)

if st.sidebar.button("⚙️ 고정밀 시뮬레이션 개시"):
    with st.spinner('고정밀 매크로 데이터 수집 및 다차원 DTW 행렬 연산을 수행 중입니다. 잠시만 기다려 주십시오...'):
        
        # 시계열 범위 확정
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=365 * lookback_years)
        
        # 7대 핵심 매크로 지표 정의
        # 나스닥, S&P500, 다우, 미국10년물국채금리, 달러인덱스, VIX, WTI원유
        macro_tickers = ['QQQ', '^GSPC', 'DIA', '^TNX', 'DX-Y.NYB', '^VIX', 'CL=F']
        all_tickers = list(set(macro_tickers + [target_stock]))
        
        # 데이터 로드
        df = fetch_comprehensive_market_data(all_tickers, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        
        # 알고리즘 구동
        top_matches, feature_weights = find_top_historical_matches(df, macro_tickers, target_stock, window_size=window, top_n=3)
        
        # --- 시각화 레이아웃 영역 ---
        st.subheader("🎯 1. 현재 시장 지배 지표 분석 (Dynamic Weights)")
        st.markdown(f"최근 {window}일 동안 **{target_stock}**의 주가 움직임과 가장 연동성이 높았던 매크로 지표별 가중치입니다. 알고리즘이 자동으로 패턴 매칭 시 이 지표들의 형상에 가중치를 부여했습니다.")
        
        # 가중치 차트 표시
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
            
            # 미래 수익률 연산
            p_curr = df[target_stock].iloc[match_idx + window]
            p_future = df[target_stock].iloc[match_idx + window + 20]
            ret = ((p_future - p_curr) / p_curr) * 100
            returns_list.append(ret)
            
            with cols[rank]:
                st.info(f"**[상위 {rank+1}위 유사 매칭 구간]**")
                st.metric("유사도 거리 점수", f"{dist_score:.2f}")
                st.write(f"📅 구간: {m_start} ~ {m_end}")
                st.metric("해당 국면 직후 20일 수익률", f"{ret:.2f}%", delta=f"{ret:.2f}%")
        
        # 종합 기대 지표 출력
        st.subheader("📊 3. 종합 통계적 모멘텀 기대치")
        avg_return = np.mean(returns_list)
        win_rate = sum(1 for r in returns_list if r > 0) / len(returns_list) * 100
        
        c1, c2, c3 = st.columns(3)
        c1.metric("앙상블 평균 예상 수익률 (20일 기대치)", f"{avg_return:.2f}%")
        c2.metric("통계적 상승 확률 (Win Rate)", f"{win_rate:.1f}%")
        c3.metric("최대 업사이드 / 다운사이드", f"{max(returns_list):.1f}% / {min(returns_list):.1f}%")
        
        # 경로 시각화 차트
        st.subheader(f"📈 4. {target_stock}의 향후 예상 주가 경로 시뮬레이션")
        path_fig = go.Figure()
        
        # 현재 패턴 선 추가
        curr_series = df[target_stock].iloc[-window:].values
        curr_norm = (curr_series - np.min(curr_series)) / (np.max(curr_series) - np.min(curr_series))
        path_fig.add_trace(go.Scatter(y=curr_norm, mode='lines', name='현재 실제 주가 경로 (최근)', line=dict(color='red', width=4)))
        
        # 과거 매칭 경로들 추가
        for rank, (match_idx, _) in enumerate(top_matches):
            # 과거 윈도우 + 미래 20일 총 경로 시각화
            past_full_series = df[target_stock].iloc[match_idx : match_idx + window + 20].values
            past_norm = (past_full_series - np.min(past_full_series[:window])) / (np.max(past_full_series[:window]) - np.min(past_full_series[:window]))
            
            path_fig.add_trace(go.Scatter(
                y=past_norm, 
                mode='lines', 
                name=f'상위 {rank+1}위 과거 시나리오 경로', 
                line=dict(dash='dash', width=2)
            ))
            
        # 기준선 표시 (현재 시점 차트 분기점)
        path_fig.add_shape(type="line", x0=window-1, y0=0, x1=window-1, y1=2, line=dict(color="black", width=2, dash="dot"))
        path_fig.add_annotation(x=window-1, y=1.5, text="현재 분석 시점 (이후는 과거의 미래 경로)", showarrow=True, argv=dict(arrowhead=1))
        
        path_fig.update_layout(title=f"{target_stock}의 국면 정규화 시나리오 프로젝션", xaxis_title="경과 일수 (Days)", yaxis_title="정규화 스케일 가격")
        st.plotly_chart(path_fig, use_container_width=True)
        
        st.success("🔒 본 리포트는 입력된 7대 거시 지표 벡터의 동적 가중치 연산 결과이며, 옥토만경님의 정밀 자산 운용 전략을 위한 정량적 기초 자료로 제공됩니다.")
