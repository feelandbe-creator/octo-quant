import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import datetime
import requests
import urllib.parse

# --- [신규] 중대 역사적 이벤트 사전 (Event Dictionary) ---
# 옥토만경님께서 필요에 따라 언제든 날짜와 사건명을 추가/수정하실 수 있습니다.
MAJOR_EVENTS = {
    "2026-02-28": "미국-이란 전쟁 시작",
    "2022-02-24": "러시아-우크라이나 전쟁 발발",
    "2020-03-11": "WHO 코로나19 팬데믹 선언",
    "2020-03-23": "연준 무제한 양적완화(QE) 발표",
    "2018-10-03": "미국 10년물 국채금리 7년 최고치 돌파 (긴축 발작)",
    "2015-08-24": "중국 위안화 쇼크 (블랙 먼데이)"
}

# --- 0. 스마트 티커 검색 및 변환 시스템 ---
def resolve_ticker(query):
    query = str(query).strip()
    if query.isdigit() and len(query) == 6:
        return f"{query}.KS"
        
    if not query.isupper() or any(ord(c) > 127 for c in query):
        encoded_query = urllib.parse.quote(query)
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={encoded_query}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
        try:
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if 'quotes' in data and len(data['quotes']) > 0:
                    return data['quotes'][0]['symbol']
        except Exception:
            pass
    return query.upper()

# --- 1. 초정밀 개별 다운로드 및 병합 엔진 ---
@st.cache_data(show_spinner=False)
def fetch_comprehensive_market_data(tickers, start_date, end_date):
    compiled_data = pd.DataFrame()
    for ticker in tickers:
        try:
            raw_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if raw_data.empty: continue
            if 'Adj Close' in raw_data.columns: price_series = raw_data['Adj Close']
            elif 'Close' in raw_data.columns: price_series = raw_data['Close']
            else: continue
            compiled_data[ticker] = price_series
        except Exception:
            continue
    compiled_data.ffill(inplace=True)
    compiled_data.dropna(inplace=True)
    return compiled_data

# --- 2. 동적 가중치 알고리즘 + 블랙스완 필터링 ---
def find_top_historical_matches(df, macro_tickers, target_stock, window_size, top_n=3):
    if target_stock not in df.columns: return [], {}
    valid_macros = [t for t in macro_tickers if t in df.columns]
    if not valid_macros: return [], {}
        
    recent_window_data = df.iloc[-window_size:]
    correlations = {}
    for ticker in valid_macros:
        corr = recent_window_data[target_stock].corr(recent_window_data[ticker])
        correlations[ticker] = abs(corr) if not np.isnan(corr) else 0.0
        
    total_corr = sum(correlations.values())
    if total_corr > 0: weights = {k: v / total_corr for k, v in correlations.items()}
    else: weights = {k: 1.0 / len(valid_macros) for k in valid_macros}
        
    macro_data = df[valid_macros]
    scaler = StandardScaler()
    scaled_macro = scaler.fit_transform(macro_data)
    weight_vector = np.array([np.sqrt(weights[ticker]) for ticker in valid_macros])
    weighted_scaled_macro = scaled_macro * weight_vector
    
    current_pattern = weighted_scaled_macro[-window_size:]
    historical_data = weighted_scaled_macro[:-window_size - 20]
    
    has_vix = '^VIX' in df.columns
    vix_data = df['^VIX'].values if has_vix else None
    
    distances = []
    for i in range(len(historical_data) - window_size):
        past_window_dates_idx = range(i, i + window_size)
        if has_vix:
            past_vix_max = np.max(vix_data[past_window_dates_idx])
            if past_vix_max > 35.0:
                continue 
        
        past_window = historical_data[i : i + window_size]
        distance, _ = fastdtw(current_pattern, past_window, dist=euclidean)
        distances.append((i, distance))
        
    distances.sort(key=lambda x: x[1])
    
    top_matches = []
    selected_indices = []
    for idx, dist in distances:
        if len(top_matches) >= top_n: break
        if any(abs(idx - s_idx) < window_size for s_idx in selected_indices): continue
        top_matches.append((idx, dist))
        selected_indices.append(idx)
        
    return top_matches, weights

# --- 3. UI/UX 대시보드 ---
st.set_page_config(page_title="옥토만경님 전용 - V3 터미널", layout="wide")
st.title("🛡️ 옥토만경님 전용: V3.2 프로페셔널 퀀트 터미널")
st.markdown("블랙스완 배제 및 중대 역사적 이벤트(전쟁, 위기) 자동 감지 시스템이 탑재되었습니다.")

st.sidebar.header("🎛️ 제어 패널")
raw_input = st.sidebar.text_input("종목명, 티커, 또는 한국 주식코드", value="JOBY")
target_stock = resolve_ticker(raw_input)
st.sidebar.markdown(f"**해석된 티커:** `{target_stock}`")

window = st.sidebar.slider("추세 분석 윈도우 (최근 N일간의 흐름)", min_value=15, max_value=90, value=45)
lookback_years = st.sidebar.slider("역사적 데이터 탐색 깊이 (년)", min_value=5, max_value=20, value=15)

if st.sidebar.button("⚙️ 고정밀 시뮬레이션 개시"):
    with st.spinner(f"'{raw_input}'(티커: {target_stock}) 데이터 수집 및 딥 매칭 중... (패닉/이벤트 감지 포함)"):
        
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=365 * lookback_years)
        
        macro_tickers = ['QQQ', '^GSPC', 'DIA', '^TNX', 'DX=F', '^VIX', 'CL=F']
        all_tickers = list(set(macro_tickers + [target_stock]))
        
        df = fetch_comprehensive_market_data(all_tickers, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        
        if target_stock not in df.columns:
            st.error(f"⚠️ '{raw_input}'에 대한 데이터를 불러오지 못했습니다. 영문 회사명이나 정확한 주식코드로 다시 시도해 주십시오.")
            st.stop()
            
        top_matches, feature_weights = find_top_historical_matches(df, macro_tickers, target_stock, window_size=window, top_n=3)
        
        if not top_matches:
            st.warning("⚠️ 지정된 과거 탐색 기간 중, VIX 35 이상의 극단적 패닉(전쟁 등) 구간을 모두 제외하고 나니 조건에 맞는 평시 데이터가 부족합니다. 탐색 기간(년)을 늘려 주십시오.")
            st.stop()
        
        st.subheader("🎯 1. 현재 시장 지배 지표 분석")
        weight_fig = go.Figure([go.Bar(
            x=[f"{k}" for k in feature_weights.keys()], 
            y=list(feature_weights.values()),
            marker_color='rgb(26, 54, 93)'
        )])
        weight_fig.update_layout(yaxis_title="가중치", xaxis_tickangle=0, height=250, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(weight_fig, use_container_width=True)
        
        st.subheader("🔮 2. 앙상블 패턴 매칭 및 시나리오")
        returns_list = []
        cols = st.columns(3)
        
        for rank, (match_idx, dist_score) in enumerate(top_matches):
            match_start_dt = df.index[match_idx]
            match_end_dt = df.index[match_idx + window]
            future_end_dt = df.index[match_idx + window + 20] # 향후 20일 예측 종료 시점
            
            m_start = match_start_dt.strftime('%Y-%m-%d')
            m_end = match_end_dt.strftime('%Y-%m-%d')
            
            p_curr = df[target_stock].iloc[match_idx + window]
            p_future = df[target_stock].iloc[match_idx + window + 20]
            ret = ((p_future - p_curr) / p_curr) * 100
            returns_list.append(ret)
            
            # [핵심 로직] 과거 매칭 구간(현재 패턴 비교 구간 + 미래 20일 예측 구간) 내에 중대 이벤트가 포함되어 있는지 검사
            event_alerts = []
            for ev_date_str, ev_name in MAJOR_EVENTS.items():
                ev_date = pd.to_datetime(ev_date_str)
                if match_start_dt <= ev_date <= future_end_dt:
                    event_alerts.append(f"[{ev_date_str}] {ev_name}")
            
            with cols[rank]:
                st.info(f"**[상위 {rank+1}위]**")
                st.write(f"📅 {m_start} ~ {m_end}")
                st.metric("거리(낮을수록 일치)", f"{dist_score:.2f}")
                st.metric("이후 20일 수익률", f"{ret:.2f}%", delta=f"{ret:.2f}%")
                
                # 이벤트가 감지되었을 경우 붉은색 경고 출력
                if event_alerts:
                    st.error("⚠️ 해당 기간 내 중대 이벤트 포함\n\n" + "\n".join(event_alerts))
        
        st.subheader("📊 3. 종합 통계적 모멘텀 기대치")
        avg_return = np.mean(returns_list)
        win_rate = sum(1 for r in returns_list if r > 0) / len(returns_list) * 100
        
        c1, c2, c3 = st.columns(3)
        c1.metric("앙상블 평균 예상 수익률", f"{avg_return:.2f}%")
        c2.metric("통계적 상승 승률", f"{win_rate:.1f}%")
        c3.metric("최대 Max / Min", f"{max(returns_list):.1f}% / {min(returns_list):.1f}%")
        
        st.subheader(f"📈 4. {target_stock}의 향후 예상 주가 경로 시뮬레이션")
        path_fig = go.Figure()
        
        curr_series = df[target_stock].iloc[-window:].values
        curr_norm = (curr_series - np.min(curr_series)) / (np.max(curr_series) - np.min(curr_series))
        path_fig.add_trace(go.Scatter(y=curr_norm, mode='lines', name='현재 실제 경로', line=dict(color='red', width=4)))
        
        for rank, (match_idx, _) in enumerate(top_matches):
            past_full_series = df[target_stock].iloc[match_idx : match_idx + window + 20].values
            past_norm = (past_full_series - np.min(past_full_series[:window])) / (np.max(past_full_series[:window]) - np.min(past_full_series[:window]))
            path_fig.add_trace(go.Scatter(y=past_norm, mode='lines', name=f'과거 {rank+1}위 시나리오', line=dict(dash='dash', width=2)))
            
        path_fig.add_shape(type="line", x0=window-1, y0=0, x1=window-1, y1=2, line=dict(color="black", width=2, dash="dot"))
        path_fig.add_annotation(x=window-1, y=1.5, text="현재 시점 (분기점)", showarrow=True, arrowhead=1)
        
        path_fig.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
            xaxis_title="경과 일수", 
            yaxis_title="정규화 스케일"
        )
        st.plotly_chart(path_fig, use_container_width=True)
        
        st.success("🔒 [블랙스완 방어 모드 가동] 과거 VIX 35 초과 구간은 탐색에서 1차 배제되었으며, 연산된 결과 내부에 특정 역사적 충격이 포함되어 있을 경우 별도의 경고가 표시됩니다.")
