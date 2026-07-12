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
        headers = {'User-Agent': 'Mozilla/5.0'}
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

# [신규 추가] 정밀 타점 계산을 위한 개별 종목 기술적 데이터 스크래핑 엔진
@st.cache_data(show_spinner=False)
def fetch_technical_indicators(ticker):
    try:
        # 최근 100일의 OHLCV 데이터 확보
        end = datetime.date.today()
        start = end - datetime.timedelta(days=100)
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty: return None
        
        close = df['Close'] if 'Close' in df.columns else df['Adj Close']
        vol = df['Volume']
        
        # 1. 이동평균선 (SMA 20, SMA 50)
        sma20 = close.rolling(window=20).mean().iloc[-1]
        sma50 = close.rolling(window=50).mean().iloc[-1]
        
        # 2. RSI (14일)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi14 = 100 - (100 / (1 + rs)).iloc[-1]
        
        # 3. 매물대 (Volume Profile - 최근 45일 기준 최대 거래량 밀집 구간)
        recent_45_df = df.iloc[-45:]
        bins = pd.cut(recent_45_df['Close' if 'Close' in df.columns else 'Adj Close'], bins=10)
        vp = recent_45_df.groupby(bins)['Volume'].sum()
        max_vol_bin = vp.idxmax()
        vp_support = max_vol_bin.mid
        
        # 4. 수급(OI 프록시) - 최근 5일 평균 거래량 vs 20일 평균 거래량 비율
        vol_5 = vol.rolling(window=5).mean().iloc[-1]
        vol_20 = vol.rolling(window=20).mean().iloc[-1]
        vol_ratio = (vol_5 / vol_20) * 100 if vol_20 > 0 else 100
        
        current_price = close.iloc[-1]
        
        return {
            'price': current_price, 'sma20': sma20, 'sma50': sma50, 
            'rsi': rsi14, 'vp_support': vp_support, 'vol_ratio': vol_ratio
        }
    except Exception:
        return None

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
st.set_page_config(page_title="AI 퀀트 터미널(옥토만경)", layout="wide")
st.header("🛡️ AI 퀀트 터미널 : V3.8(옥토만경)")
st.markdown("정량화된 실전 매매 지침 및 궤적 추적 엔진 탑재")
st.markdown("---")
st.subheader("🎛️ 제어 패널")

col1, col2 = st.columns(2)
with col1:
    raw_input = st.text_input("종목명, 티커, 또는 한국 주식코드", value="JOBY")
    target_stock = resolve_ticker(raw_input)
    st.caption(f"**해석된 티커:** `{target_stock}`")
    window = st.selectbox("추세 분석 윈도우 (최근 N일간의 흐름)", options=[15, 30, 45, 60, 90], index=2) 
with col2:
    top_n_input = st.selectbox("유사 국면 매칭 개수 (N)", options=[3, 4, 5, 6, 7], index=2) 
    lookback_years = st.selectbox("역사적 데이터 탐색 깊이 (년)", options=[5, 8, 10, 12, 15, 20], index=4) 

st.markdown("<br>", unsafe_allow_html=True)

if st.button("⚙️ 시뮬레이션 시작", use_container_width=True):
    with st.spinner(f"'{raw_input}' 데이터 연산 및 AI 실전 매매 판독 중..."):
        
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=365 * lookback_years)
        # ^GSPC(S&P500)는 거시 선물 추이 프록시로 사용됨
        macro_tickers = ['QQQ', '^GSPC', 'DIA', '^TNX', 'DX=F', '^VIX', 'CL=F']
        all_tickers = list(set(macro_tickers + [target_stock]))
        
        df = fetch_comprehensive_market_data(all_tickers, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        tech_data = fetch_technical_indicators(target_stock)
        
        if target_stock not in df.columns:
            st.error(f"⚠️ '{raw_input}' 데이터를 수신하지 못했습니다.")
            st.stop()
            
        top_matches, feature_weights, top_excluded = find_top_historical_matches(df, macro_tickers, target_stock, window_size=window, top_n=top_n_input)
        
        st.markdown("---")
        
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
        past_slopes = [] # 궤적 이탈 추적용 (과거 시나리오의 마지막 5일 기울기)
        
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
                
                # 궤적 연산 (정규화)
                past_full_series = df[target_stock].iloc[match_idx : match_idx + window + 20].values
                past_norm = (past_full_series - np.min(past_full_series[:window])) / (np.max(past_full_series[:window]) - np.min(past_full_series[:window]))
                past_slopes.append(past_norm[window-1] - past_norm[window-5]) # 과거 5일간의 모멘텀 기울기
                
                with cols[rank]:
                    st.info(f"**[클린 상위 {actual_rank}위]**")
                    st.write(f"📅 {m_start} ~ {m_end}")
                    st.metric("거리", f"{dist_score:.2f}")
                    st.metric("이후 20일 수익률", f"{ret:.2f}%", delta=f"{ret:.2f}%")
        
        # --- 📊 3. 종합 통계적 모멘텀 기대치 및 실전 매매 행동 지침 ---
        st.subheader("📊 3. 종합 통계적 모멘텀 기대치 및 실전 매매 지침")
        avg_return = np.mean(returns_list)
        win_rate = sum(1 for r in returns_list if r > 0) / len(returns_list) * 100
        max_ret = max(returns_list)
        min_ret = min(returns_list)
        avg_dist = np.mean(distances_list)
        
        # 3-1. 종합 기대치 시그널
        signal_text, signal_color, reasoning = "", "", ""
        risk_reward_ratio = max_ret / abs(min_ret) if min_ret < 0 else float('inf')
        
        if win_rate == 100 and avg_return >= 5.0:
            signal_text, signal_color = f"적극 매수 (향후 20일 {avg_return:.2f}% 상승 예상)", "#FF2A2A"
            reasoning = "과거 전 구간 100% 상승 및 기대수익률 +5% 이상의 최상급 A급 진입 찬스입니다."
        elif win_rate == 100 and avg_return > 0 and min_ret >= -1.5:
            signal_text, signal_color = f"매수 고려 (향후 20일 {avg_return:.2f}% 상승 예상)", "#FF4B4B"
            reasoning = "예상 수익률은 낮으나 100% 승률과 극도로 제한된 하방(최대 -1.5% 이내)이 보장된 안전 진입 구간입니다."
        elif win_rate >= 66 and risk_reward_ratio >= 2.0:
            signal_text, signal_color = f"매수 고려 (향후 20일 {avg_return:.2f}% 상승 예상)", "#FF4B4B"
            reasoning = "승률 66% 이상 및 손익비(하락대비 상승폭) 2배 이상의 정석적인 퀀트 매수 구간입니다."
        elif avg_return > 0 and win_rate < 50:
            signal_text, signal_color = "관망 (통계적 왜곡 리스크)", "#777777"
            reasoning = "평균은 양수이나 승률이 절반 미만인 전형적인 소수 폭등 착시(High Risk) 구간입니다."
        elif avg_return <= 0 and win_rate > 33:
            signal_text, signal_color = "관망 (방향성 부재)", "#777777"
            reasoning = "승률과 기대 수익률 모두 통계적 우위를 점하지 못한 중립 구간입니다."
        elif avg_return < 0 and win_rate <= 33 and min_ret >= -5.0:
            signal_text, signal_color = f"매도 고려 (향후 20일 {abs(avg_return):.2f}% 하락 예상)", "#1C83E1"
            reasoning = "낮은 승률과 평균치 하락이 예상되므로 비중 축소가 권장됩니다."
        else: 
            signal_text, signal_color = f"적극 매도 (향후 20일 {abs(avg_return):.2f}% 하락 예상)", "#0055FF"
            reasoning = "강력한 하방 압력 및 바닥권 승률이 겹친 구간이므로 즉각적인 하방 리스크 회피가 필요합니다."

        st.markdown(f"<div style='border:2px solid {signal_color}; border-radius:10px; padding:20px; background-color:rgba(255,255,255,0.05);'>"
                    f"<h2 style='color:{signal_color}; margin-top:0px; text-align:center;'>{signal_text}</h2>"
                    f"<p style='font-size:16px; margin-bottom:0px; text-align:center;'><strong>[판단 근거]</strong> {reasoning}</p></div><br>", unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("앙상블 평균 예상 수익률", f"{avg_return:.2f}%")
        c2.metric("통계적 상승 승률", f"{win_rate:.1f}%")
        c3.metric("최대 Max / Min", f"{max_ret:.1f}% / {min_ret:.1f}%")

        st.markdown("---")
        st.subheader("💡 정량적 실전 매매 행동 지침 (Numerical Action Plan)")
        
        # 1. 자금 투입 비중 (거리 점수 연계)
        alloc_level = ""
        alloc_ratio = ""
        # 거리 기준: 통상적으로 15 이하면 매우 가까움, 25 이상이면 멈. (유동적 데이터스케일 감안)
        if avg_dist < 15.0:
            alloc_level, alloc_ratio = "공격적", "70% 이상"
        elif 15.0 <= avg_dist < 25.0:
            alloc_level, alloc_ratio = "중립적", "50% 전후"
        else:
            alloc_level, alloc_ratio = "보수적", "30% 이하"
            
        st.markdown(f"**① 자금 투입 비중: <span style='color:#FF4B4B;'>{alloc_level} 진입 ({alloc_ratio} 할당)</span>**", unsafe_allow_html=True)
        st.markdown(f"↳ **근거:** 현재 도출된 상위 국면들의 평균 유사도 거리 점수가 **{avg_dist:.2f}**로 산출되었습니다. "
                    f"거리가 15 미만이면 과거 패턴과의 동조화가 극대화된 공격적 구간이며, 25 이상이면 오차 가능성을 대비해 비중을 30% 이하로 통제해야 합니다.")
        
        # 2. 진입 타점 정밀화 (기술적 지표 5개 연계)
        if tech_data:
            p_price = tech_data['price']
            st.markdown(f"**② 진입 타점 정밀화 (현재가: {p_price:.2f})**")
            
            support_line = tech_data['sma20'] if p_price > tech_data['sma20'] else tech_data['sma50']
            rsi_stat = "과매도(매수 유리)" if tech_data['rsi'] < 40 else "과매수(조정 주의)" if tech_data['rsi'] > 60 else "중립"
            vol_stat = "기관/외인 수급 폭발적 유입 추정" if tech_data['vol_ratio'] > 150 else "평이한 거래량 유지 중"
            
            # S&P500의 최근 5일 모멘텀을 선물 추세로 대용
            macro_trend = df['^GSPC'].iloc[-1] - df['^GSPC'].iloc[-5]
            macro_stat = "상승 추세(위험자산 선호)" if macro_trend > 0 else "하락 추세(위험자산 회피)"
            
            st.markdown(
                f"- **이동평균선 주요 지지선:** **{support_line:.2f}** 부근 (돌파 시 강력한 지지대 역할 기대)\n"
                f"- **RSI (14일) 지표:** **{tech_data['rsi']:.1f}** ({rsi_stat})\n"
                f"- **최대 매물대 (Volume Profile):** 최근 45일간 최대 거래 밀집 구간인 **{tech_data['vp_support']:.2f}** 방어 여부 확인\n"
                f"- **수급/OI 현황 프록시:** 최근 5일 거래량이 20일 평균 대비 **{tech_data['vol_ratio']:.1f}%** 발생 ({vol_stat})\n"
                f"- **거시 선물 추이:** 글로벌 증시 벤치마크(S&P500) 기준 최근 5일 **{macro_stat}**\n"
                f"↳ **종합 타점 제언:** 거시 시그널이 긍정적일 경우, 현재가 추격 매수보다는 **{support_line:.2f} (이평선 지지)**와 **{tech_data['vp_support']:.2f} (최대 매물대)** 사이의 밴드에서 분할 매수 진입을 권장합니다."
            )
        else:
            st.markdown("**② 진입 타점 정밀화:** 해당 종목의 실시간 기술적 지표를 불러올 수 없습니다.")

        # 3. 청산 전략 (궤적 이탈 검사)
        curr_series = df[target_stock].iloc[-window:].values
        curr_norm = (curr_series - np.min(curr_series)) / (np.max(curr_series) - np.min(curr_series))
        curr_slope = curr_norm[-1] - curr_norm[-5] # 현재 경로 최근 5일 정규화 기울기
        avg_past_slope = np.mean(past_slopes)      # 과거 시나리오 동일 시점 평균 기울기
        
        exit_signal, exit_color, exit_reason = "", "", ""
        slope_diff = curr_slope - avg_past_slope
        
        # 궤적 이탈 임계치(Threshold) 검사
        if slope_diff >= -0.05:
            exit_signal, exit_color = "매수 유지", "#FF4B4B"
            exit_reason = f"현재 경로의 최근 5일 상승 탄력({curr_slope:+.3f})이 과거 시나리오의 평균 궤적({avg_past_slope:+.3f})을 정상적으로 추종하거나 오히려 상회하고 있습니다. 20일 청산 시점까지 포지션을 강력히 유지하십시오."
        elif -0.15 <= slope_diff < -0.05:
            exit_signal, exit_color = "관망 (경고)", "#777777"
            exit_reason = f"현재 경로의 상승 탄력({curr_slope:+.3f})이 과거 평균 궤적({avg_past_slope:+.3f})보다 둔화되며 하단으로 미세 이탈 중입니다. 신규 매수는 보류하고 지지선 붕괴 여부를 관망하십시오."
        else:
            exit_signal, exit_color = "매도 (이탈 확정)", "#0055FF"
            exit_reason = f"현재 경로의 모멘텀({curr_slope:+.3f})이 과거 궤적({avg_past_slope:+.3f})을 심각하게 하향 이탈했습니다. 과거의 성공 시나리오가 무효화되었으므로 20일을 기다리지 말고 즉각 익절/손절 청산을 집행하십시오."
            
        st.markdown(f"**③ 청산 전략 (실시간 궤적 추적): <span style='color:{exit_color};'>{exit_signal}</span>**", unsafe_allow_html=True)
        st.markdown(f"↳ **근거:** {exit_reason}")

        # --- 📈 4. 예상 주가 경로 시뮬레이션 ---
        st.subheader(f"📈 4. {target_stock}의 향후 예상 주가 경로 시뮬레이션")
        path_fig = go.Figure()
        
        path_fig.add_trace(go.Scatter(y=curr_norm, mode='lines', name='현재 실제 경로', line=dict(color='red', width=4)))
        
        for rank, (match_idx, _) in enumerate(top_matches):
            past_full_series = df[target_stock].iloc[match_idx : match_idx + window + 20].values
            past_norm = (past_full_series - np.min(past_full_series[:window])) / (np.max(past_full_series[:window]) - np.min(past_full_series[:window]))
            path_fig.add_trace(go.Scatter(y=past_norm, mode='lines', name=f'클린 과거 {rank+1}위 시나리오', line=dict(width=2)))
            
        path_fig.add_shape(type="line", x0=window-1, y0=0, x1=window-1, y1=2, line=dict(color="black", width=2, dash="dot"))
        path_fig.add_annotation(x=window-1, y=1.5, text="현재 시점", showarrow=True, arrowhead=1)
        path_fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5), xaxis_title="경과 일수", yaxis_title="정규화 스케일")
        st.plotly_chart(path_fig, use_container_width=True)
