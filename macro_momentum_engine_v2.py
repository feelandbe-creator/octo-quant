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

# --- 0. 중대 역사적 이벤트 사전 ---
MAJOR_EVENTS = {
    "2026-02-28": "미국-이란 전쟁 시작",
    "2022-02-24": "러시아-우크라이나 전쟁 발발",
    "2020-03-11": "WHO 코로나19 팬데믹 선언",
    "2020-03-23": "연준 무제한 양적완화(QE) 발표",
    "2018-10-03": "미국 10년물 국채금리 7년 최고치 돌파",
    "2015-08-24": "중국 위안화 쇼크 (블랙 먼데이)"
}

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

# --- 2. 클린 구간 추출 알고리즘 ---
def find_top_historical_matches(df, macro_tickers, target_stock, window_size, top_n=5):
    if target_stock not in df.columns: return [], {}, []
    valid_macros = [t for t in macro_tickers if t in df.columns]
    if not valid_macros: return [], {}, []
        
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
    
    clean_distances = []
    excluded_distances = []
    
    for i in range(len(historical_data) - window_size):
        match_start_dt = df.index[i]
        future_end_dt = df.index[i + window_size + 20]
        
        if has_vix:
            past_window_dates_idx = range(i, i + window_size)
            past_vix_max = np.max(vix_data[past_window_dates_idx])
            if past_vix_max > 35.0:
                continue 
        
        event_alerts = []
        for ev_date_str, ev_name in MAJOR_EVENTS.items():
            ev_date = pd.to_datetime(ev_date_str)
            if match_start_dt <= ev_date <= future_end_dt:
                event_alerts.append(f"[{ev_date_str}] {ev_name}")
        
        past_window = historical_data[i : i + window_size]
        distance, _ = fastdtw(current_pattern, past_window, dist=euclidean)
        
        if event_alerts:
            excluded_distances.append((i, distance, event_alerts))
        else:
            clean_distances.append((i, distance))
            
    clean_distances.sort(key=lambda x: x[1])
    top_matches = []
    selected_indices = []
    for idx, dist in clean_distances:
        if len(top_matches) >= top_n: break
        if any(abs(idx - s_idx) < window_size for s_idx in selected_indices): continue
        top_matches.append((idx, dist))
        selected_indices.append(idx)
        
    excluded_distances.sort(key=lambda x: x[1])
    top_excluded = excluded_distances[:2]
        
    return top_matches, weights, top_excluded

# --- 3. UI/UX 대시보드 ---
st.set_page_config(page_title="옥토만경님 전용 - V3.6 AI 터미널", layout="wide")
st.title("🛡️ 옥토만경님 전용: V3.6 인공지능 퀀트 터미널")
st.markdown("다차원 매크로 매칭 및 5단계 실전 매매 의사결정 엔진이 탑재되었습니다.")

st.sidebar.header("🎛️ 제어 패널")
raw_input = st.sidebar.text_input("종목명, 티커, 또는 한국 주식코드", value="JOBY")
target_stock = resolve_ticker(raw_input)
st.sidebar.markdown(f"**해석된 티커:** `{target_stock}`")

window = st.sidebar.slider("추세 분석 윈도우 (최근 N일간의 흐름)", min_value=15, max_value=90, value=45)
top_n_input = st.sidebar.slider("유사 국면 매칭 개수 (N)", min_value=3, max_value=7, value=5)
lookback_years = st.sidebar.slider("역사적 데이터 탐색 깊이 (년)", min_value=5, max_value=20, value=15)

if st.sidebar.button("⚙️ 고정밀 시뮬레이션 개시"):
    with st.spinner(f"'{raw_input}' 데이터 연산 및 인공지능 매매 판독 중..."):
        
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=365 * lookback_years)
        macro_tickers = ['QQQ', '^GSPC', 'DIA', '^TNX', 'DX=F', '^VIX', 'CL=F']
        all_tickers = list(set(macro_tickers + [target_stock]))
        
        df = fetch_comprehensive_market_data(all_tickers, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        
        if target_stock not in df.columns:
            st.error(f"⚠️ '{raw_input}' 데이터를 수신하지 못했습니다.")
            st.stop()
            
        top_matches, feature_weights, top_excluded = find_top_historical_matches(df, macro_tickers, target_stock, window_size=window, top_n=top_n_input)
        
        if top_excluded:
            st.subheader("🚫 대외 돌발 변수 격리 안내")
            for idx, dist, alerts in top_excluded:
                ex_start = df.index[idx].strftime('%Y-%m-%d')
                ex_end = df.index[idx + window].strftime('%Y-%m-%d')
                event_str = ", ".join(alerts)
                st.error(f"⚠️ 과거 구간 [{ex_start} ~ {ex_end}] 내에 **{event_str}**이 포함되어 있습니다. 이 이벤트 때문에 우리의 결과 종합통계적 모멘텀 기대치에는 반영하지 않겠습니다.")
        
        if not top_matches:
            st.warning("⚠️ 클린 데이터가 부족합니다. 탐색 깊이를 늘려주십시오.")
            st.stop()
            
        st.subheader("🎯 1. 현재 시장 지배 지표 분석")
        weight_fig = go.Figure([go.Bar(x=list(feature_weights.keys()), y=list(feature_weights.values()), marker_color='rgb(26, 54, 93)')])
        weight_fig.update_layout(yaxis_title="가중치", height=230, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(weight_fig, use_container_width=True)
        
        st.subheader(f"🔮 2. 앙상블 패턴 매칭 및 시나리오 (보충된 클린 구간 Top {len(top_matches)})")
        returns_list = []
        distances_list = []
        
        for i in range(0, len(top_matches), 3):
            chunk = top_matches[i:i+3]
            cols = st.columns(len(chunk))
            for rank, (match_idx, dist_score) in enumerate(chunk):
                actual_rank = i + rank + 1
                m_start = df.index[match_idx].strftime('%Y-%m-%d')
                m_end = df.index[match_idx + window].strftime('%Y-%m-%d')
                p_curr = df[target_stock].iloc[match_idx + window]
                p_future = df[target_stock].iloc[match_idx + window + 20]
                ret = ((p_future - p_curr) / p_curr) * 100
                returns_list.append(ret)
                distances_list.append(dist_score)
                
                with cols[rank]:
                    st.info(f"**[클린 상위 {actual_rank}위]**")
                    st.write(f"📅 {m_start} ~ {m_end}")
                    st.metric("거리(낮을수록 일치)", f"{dist_score:.2f}")
                    st.metric("이후 20일 수익률", f"{ret:.2f}%", delta=f"{ret:.2f}%")
        
        # --- 📊 3. 종합 통계적 모멘텀 기대치 및 AI 매매 시그널 판독 ---
        st.subheader("📊 3. 종합 통계적 모멘텀 기대치 및 실전 매매 지침")
        avg_return = np.mean(returns_list)
        win_rate = sum(1 for r in returns_list if r > 0) / len(returns_list) * 100
        max_ret = max(returns_list)
        min_ret = min(returns_list)
        avg_dist = np.mean(distances_list)
        
        # 리스크-리워드(손익비) 계산: 하락폭 대비 상승폭이 몇 배인지 (단, 하락이 없으면 무한대)
        risk_reward_ratio = max_ret / abs(min_ret) if min_ret < 0 else float('inf')
        
        # 5단계 의사결정 알고리즘
        signal_text = ""
        signal_color = ""
        reasoning = ""
        
        if win_rate == 100 and avg_return >= 5.0:
            signal_text = f"적극 매수 (향후 20거래일 동안 {avg_return:.2f}% 상승 예상)"
            signal_color = "#FF2A2A" # 강렬한 빨강
            reasoning = f"과거 {len(top_matches)}번의 유사 국면에서 단 한 번의 예외도 없이 모두 상승(승률 100%)했으며, 예상 수익률이 +5% 이상인 **[최상의 A급 진입 찬스]**입니다. 포트폴리오 내 투자 비중을 과감하게 늘리는 전략이 유효합니다."
            
        elif win_rate == 100 and avg_return > 0 and min_ret >= -1.5:
            signal_text = f"매수 고려 (향후 20거래일 동안 {avg_return:.2f}% 상승 예상)"
            signal_color = "#FF4B4B" # 빨강
            reasoning = f"예상 수익률({avg_return:.2f}%)은 크지 않으나, 승률이 100%이며 최악의 경우에도 하락폭({min_ret:.2f}%)이 극히 제한적인 **[하방 경직성 보장 구간]**입니다. 손실 가능성이 낮으므로 안전 지향형 진입에 적합합니다."
            
        elif win_rate >= 66 and risk_reward_ratio >= 2.0:
            signal_text = f"매수 고려 (향후 20거래일 동안 {avg_return:.2f}% 상승 예상)"
            signal_color = "#FF4B4B" # 빨강
            reasoning = f"매수 최우선 지표인 승률이 66% 이상({win_rate:.1f}%)이며, 하락폭 대비 기대 상승폭(손익비)이 2배 이상 확보된 정석적인 퀀트 진입 구간입니다."
            
        elif avg_return > 0 and win_rate < 50:
            signal_text = "관망 (통계적 왜곡 리스크)"
            signal_color = "#777777" # 회색
            reasoning = f"예상 평균 수익률은 {avg_return:.2f}%로 양수이나, 실제 승률은 {win_rate:.1f}%에 불과합니다. 이는 소수의 극단적 상승(착시 효과)이 평균값을 오염시킨 전형적인 **[모 아니면 도(High Risk)]** 국면이므로 진입을 엄격히 제한합니다."
            
        elif avg_return <= 0 and win_rate > 33:
            signal_text = "관망 (방향성 부재)"
            signal_color = "#777777"
            reasoning = f"승률({win_rate:.1f}%)과 기대 수익률({avg_return:.2f}%) 모두 매매를 집행할 만한 통계적 우위를 확보하지 못했습니다. 추세가 명확해질 때까지 관망하십시오."
            
        elif avg_return < 0 and win_rate <= 33 and min_ret >= -5.0:
            signal_text = f"매도 고려 (향후 20거래일 동안 {abs(avg_return):.2f}% 하락 예상)"
            signal_color = "#1C83E1" # 파랑
            reasoning = f"승률이 현저히 낮고 평균 수익률이 음수 구간에 진입했습니다. 하방 리스크가 열려 있으므로 비중 축소 및 매도를 검토해야 합니다."
            
        else: # avg_return < 0 and min_ret < -5.0
            signal_text = f"적극 매도 (향후 20거래일 동안 {abs(avg_return):.2f}% 하락 예상)"
            signal_color = "#0055FF" # 짙은 파랑
            reasoning = f"유사 국면들의 승률({win_rate:.1f}%)이 바닥권이며 강력한 하방 압력(최악의 경우 {min_ret:.2f}%)이 예상됩니다. 즉각적인 포지션 정리 및 하방 리스크 회피가 최우선입니다."

        # 판독 결과 출력
        st.markdown(f"<div style='border:2px solid {signal_color}; border-radius:10px; padding:20px; background-color:rgba(255,255,255,0.05);'>"
                    f"<h2 style='color:{signal_color}; margin-top:0px; text-align:center;'>{signal_text}</h2>"
                    f"<p style='font-size:16px; margin-bottom:0px;'><strong>[판단 근거]</strong> {reasoning}</p>"
                    f"</div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("앙상블 평균 예상 수익률", f"{avg_return:.2f}%")
        c2.metric("통계적 상승 승률", f"{win_rate:.1f}%")
        c3.metric("최대 Max / Min", f"{max_ret:.1f}% / {min_ret:.1f}%")

        # 옥토만경님 전용 행동 지침 (투입 비중 및 타점 가이드)
        st.info(f"💡 **[실전 매매 행동 지침]**\n"
                f"- **자금 투입 비중 (거리 점수 연계):** 현재 평균 거리 점수는 **{avg_dist:.2f}**입니다. 거리가 낮을수록(패턴 일치도가 높을수록) 할당 시드머니의 70~80%까지 공격적 진입이 가능하며, 거리가 높다면 30% 이하로 비중을 통제하십시오.\n"
                f"- **진입 타점 정밀화:** 거시 지표로 방향이 정해졌다면, 실제 매수는 이동평균선 주요 지지선이나 RSI 단기 과매도 권역 등 기술적 지표로 방아쇠를 당기십시오.\n"
                f"- **청산 전략:** 매수 후 약 20거래일이 도래하면 포지션을 정리하는 것을 원칙으로 하며, 아래 시뮬레이션의 현재 경로(빨간선)가 과거 시나리오(실선)를 엇나가기 시작하면 즉각 대응하십시오.")
        
        # --- 📈 4. 예상 주가 경로 시뮬레이션 (완벽한 실선 적용) ---
        st.subheader(f"📈 4. {target_stock}의 향후 예상 주가 경로 시뮬레이션")
        path_fig = go.Figure()
        curr_series = df[target_stock].iloc[-window:].values
        curr_norm = (curr_series - np.min(curr_series)) / (np.max(curr_series) - np.min(curr_series))
        
        path_fig.add_trace(go.Scatter(y=curr_norm, mode='lines', name='현재 실제 경로', line=dict(color='red', width=4)))
        
        for rank, (match_idx, _) in enumerate(top_matches):
            past_full_series = df[target_stock].iloc[match_idx : match_idx + window + 20].values
            past_norm = (past_full_series - np.min(past_full_series[:window])) / (np.max(past_full_series[:window]) - np.min(past_full_series[:window]))
            
            # 과거 시나리오 점선(dash) 제거 완료 -> 뚜렷한 실선(width=2)으로 렌더링
            path_fig.add_trace(go.Scatter(y=past_norm, mode='lines', name=f'클린 과거 {rank+1}위 시나리오', line=dict(width=2)))
            
        path_fig.add_shape(type="line", x0=window-1, y0=0, x1=window-1, y1=2, line=dict(color="black", width=2, dash="dot"))
        path_fig.add_annotation(x=window-1, y=1.5, text="현재 시점", showarrow=True, arrowhead=1)
        path_fig.update_layout(margin=dict(l=10,
