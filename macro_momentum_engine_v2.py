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
            if 'Adj Close' in raw_data.columns: price_series = raw_data['Adj Close'].squeeze()
            elif 'Close' in raw_data.columns: price_series = raw_data['Close'].squeeze()
            else: continue
            compiled_data[ticker] = price_series
        except Exception:
            continue
    compiled_data.ffill(inplace=True)
    compiled_data.dropna(inplace=True)
    return compiled_data

@st.cache_data(show_spinner=False)
def fetch_technical_indicators(ticker):
    try:
        end = datetime.date.today()
        start = end - datetime.timedelta(days=100)
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty: return None
        
        close = df['Close'].squeeze() if 'Close' in df.columns else df['Adj Close'].squeeze()
        vol = df['Volume'].squeeze() if 'Volume' in df.columns else pd.Series(0, index=close.index)
        
        sma20 = float(close.rolling(window=20).mean().iloc[-1])
        sma50 = float(close.rolling(window=50).mean().iloc[-1])
        
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi14 = float(100 - (100 / (1 + rs)).iloc[-1])
        
        recent_45_close = close.iloc[-45:]
        recent_45_vol = vol.iloc[-45:]
        bins = pd.cut(recent_45_close, bins=10)
        vp = recent_45_vol.groupby(bins, observed=False).sum() 
        max_vol_bin = vp.idxmax()
        vp_support = float(max_vol_bin.mid)
        
        vol_5 = float(vol.rolling(window=5).mean().iloc[-1])
        vol_20 = float(vol.rolling(window=20).mean().iloc[-1])
        vol_ratio = float((vol_5 / vol_20) * 100) if vol_20 > 0 else 100.0
        
        current_price = float(close.iloc[-1])
        
        opt_data = None
        try:
            tk = yf.Ticker(ticker)
            opts_dates = tk.options
            if opts_dates:
                nearest_date = opts_dates[0] 
                chain = tk.option_chain(nearest_date)
                calls = chain.calls
                puts = chain.puts
                max_call = calls.loc[calls['openInterest'].idxmax()] if not calls.empty else None
                max_put = puts.loc[puts['openInterest'].idxmax()] if not puts.empty else None
                opt_data = {
                    'expiry': nearest_date,
                    'call_strike': float(max_call['strike']) if max_call is not None else 0.0,
                    'call_oi': int(max_call['openInterest']) if max_call is not None else 0,
                    'call_vol': int(max_call['volume']) if max_call is not None else 0,
                    'put_strike': float(max_put['strike']) if max_put is not None else 0.0,
                    'put_oi': int(max_put['openInterest']) if max_put is not None else 0,
                    'put_vol': int(max_put['volume']) if max_put is not None else 0,
                }
        except Exception:
            pass 
            
        return {
            'price': current_price, 'sma20': sma20, 'sma50': sma50, 
            'rsi': rsi14, 'vp_support': vp_support, 'vol_ratio': vol_ratio,
            'opt_data': opt_data
        }
    except Exception:
        return None

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

# --- 3. UI/UX 대시보드 (다크 테마 유니폼 적용) ---
st.set_page_config(page_title="AI 퀀트 터미널(옥토만경)", layout="wide", initial_sidebar_state="collapsed")

# [수정 완료] 타이틀 폰트 크기 축소 및 중요 숫자 강조용 색상(황금색) CSS 추가
st.markdown("""
<style>
    .report-title { font-size: 22px; font-weight: 800; color: #F9FAFB; margin-bottom: 0px; }
    .report-subtitle { font-size: 16px; color: #9CA3AF; margin-bottom: 30px; border-bottom: 2px solid #374151; padding-bottom: 10px; }
    .section-header { font-size: 22px; font-weight: 700; color: #F9FAFB; margin-top: 40px; margin-bottom: 15px; border-left: 4px solid #3B82F6; padding-left: 10px; }
    .metric-card { background-color: rgba(255,255,255,0.03); padding: 15px; border-radius: 8px; border: 1px solid #374151; margin-bottom: 15px; color: #E5E7EB; }
    .highlight-red { color: #EF4444; font-weight: 700; }
    .highlight-blue { color: #3B82F6; font-weight: 700; }
    .highlight-num { color: #FBBF24; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="report-title">🛡️ AI 퀀트 터미널 : V4.2 (옥토만경)</div>', unsafe_allow_html=True)
st.markdown('<div class="report-subtitle">실시간 정량적 실전 매매 지침 및 파생(OI) 수급 추적 엔진 탑재 리포트</div>', unsafe_allow_html=True)

# [수정 완료] 제어 패널 명칭 간소화
with st.expander("🎛️ 분석 설정", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        raw_input = st.text_input("종목명, 티커, 또는 한국 주식코드", value="JOBY")
        target_stock = resolve_ticker(raw_input)
        st.caption(f"**해석된 티커:** `{target_stock}`")
        window = st.selectbox("추세 분석 윈도우 (최근 N일간의 흐름)", options=[15, 30, 45, 60, 90], index=2) 
    with col2:
        top_n_input = st.selectbox("유사 국면 매칭 개수 (N)", options=[3, 4, 5, 6, 7], index=2) 
        lookback_years = st.selectbox("역사적 데이터 탐색 깊이 (년)", options=[5, 8, 10, 12, 15, 20], index=4) 

    run_sim = st.button("⚙️ 시뮬레이션 시작", use_container_width=True, type="primary")

if run_sim:
    with st.spinner(f"'{raw_input}' 정밀 데이터 스크래핑 및 인공지능 매매 판독 중..."):
        
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=365 * lookback_years)
        macro_tickers = ['QQQ', '^GSPC', 'DIA', '^TNX', 'DX=F', '^VIX', 'CL=F']
        all_tickers = list(set(macro_tickers + [target_stock]))
        
        df = fetch_comprehensive_market_data(all_tickers, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        tech_data = fetch_technical_indicators(target_stock)
        
        if target_stock not in df.columns:
            st.error(f"⚠️ '{raw_input}' 데이터를 수신하지 못했습니다. 상장 폐지되었거나 티커가 올바르지 않습니다.")
            st.stop()
            
        top_matches, feature_weights, top_excluded = find_top_historical_matches(df, macro_tickers, target_stock, window_size=window, top_n=top_n_input)
        
        if top_excluded:
            st.warning("⚠️ **[시스템 알림] 대외 돌발 변수 격리 조치 완료**")
            for idx, dist, alerts in top_excluded:
                ex_start = df.index[idx].strftime('%Y-%m-%d')
                ex_end = df.index[idx + window].strftime('%Y-%m-%d')
                event_str = ", ".join(alerts)
                st.caption(f" └ 과거 구간 [{ex_start} ~ {ex_end}] 내에 **{event_str}**이 포함되어 있어 통계에서 원천 배제되었습니다.")
        
        if not top_matches:
            st.error("⚠️ 클린 데이터가 부족합니다. 탐색 깊이를 늘려주십시오.")
            st.stop()

        utc_now = datetime.datetime.utcnow()
        kst_now = (utc_now + datetime.timedelta(hours=9)).strftime("%Y-%m-%d %H:%M:%S")
        edt_now = (utc_now - datetime.timedelta(hours=4)).strftime("%Y-%m-%d %H:%M:%S")

        st.markdown('<div class="section-header">📊 1. 정량적 실전 매매 행동 지침 (Real-time Action Plan)</div>', unsafe_allow_html=True)
        st.markdown(f"*(실시간 데이터 기준 시각: **KST** {kst_now} / **뉴욕** {edt_now})*")
        
        returns_list, distances_list, past_slopes = [], [], []
        for match_idx, dist_score in top_matches:
            p_curr = df[target_stock].iloc[match_idx + window]
            p_future = df[target_stock].iloc[match_idx + window + 20]
            returns_list.append(((p_future - p_curr) / p_curr) * 100)
            distances_list.append(dist_score)
            
            past_full = df[target_stock].iloc[match_idx : match_idx + window + 20].values
            past_norm = (past_full - np.min(past_full[:window])) / (np.max(past_full[:window]) - np.min(past_full[:window]))
            past_slopes.append(past_norm[window-1] - past_norm[window-5])
            
        avg_return = np.mean(returns_list)
        win_rate = sum(1 for r in returns_list if r > 0) / len(returns_list) * 100
        max_ret = max(returns_list)
        min_ret = min(returns_list)
        avg_dist = np.mean(distances_list)
        risk_reward_ratio = max_ret / abs(min_ret) if min_ret < 0 else float('inf')
        
        signal_text, signal_color, reasoning = "", "", ""
        if win_rate == 100 and avg_return >= 5.0:
            signal_text, signal_color = f"적극 매수 (향후 20일 {avg_return:.2f}% 상승 예상)", "#EF4444"
            reasoning = "과거 전 구간 100% 상승 및 기대수익률 +5% 이상의 최상급 A급 진입 찬스입니다."
        elif win_rate == 100 and avg_return > 0 and min_ret >= -1.5:
            signal_text, signal_color = f"매수 고려 (향후 20일 {avg_return:.2f}% 상승 예상)", "#EF4444"
            reasoning = "수익률은 낮으나 100% 승률과 극도로 제한된 하방 리스크가 보장된 안전 진입 구간입니다."
        elif win_rate >= 66 and risk_reward_ratio >= 2.0:
            signal_text, signal_color = f"매수 고려 (향후 20일 {avg_return:.2f}% 상승 예상)", "#EF4444"
            reasoning = "승률 66% 이상 및 손익비 2배 이상을 충족하는 정석적인 퀀트 매수 구간입니다."
        elif avg_return > 0 and win_rate < 50:
            signal_text, signal_color = "관망 (통계적 왜곡 리스크)", "#9CA3AF"
            reasoning = "평균은 양수이나 승률이 절반 미만인 착시 효과(소수 폭등) 구간이므로 진입을 보류하십시오."
        elif avg_return <= 0 and win_rate > 33:
            signal_text, signal_color = "관망 (방향성 부재)", "#9CA3AF"
            reasoning = "승률과 기대 수익률 모두 통계적 우위를 점하지 못한 중립 구간입니다."
        elif avg_return < 0 and win_rate <= 33 and min_ret >= -5.0:
            signal_text, signal_color = f"매도 고려 (향후 20일 {abs(avg_return):.2f}% 하락 예상)", "#3B82F6"
            reasoning = "낮은 승률과 평균치 하락이 예상되므로 비중 축소가 권장됩니다."
        else: 
            signal_text, signal_color = f"적극 매도 (향후 20일 {abs(avg_return):.2f}% 하락 예상)", "#2563EB"
            reasoning = "강력한 하방 압력 및 바닥권 승률이 겹친 구간이므로 즉각적인 포지션 정리가 필요합니다."

        # [수정 완료] 시그널 박스 색상을 다크 테마에 맞게 튜닝 (반투명 배경, 어두운 테두리)
        st.markdown(f"""
        <div style='border-left: 5px solid {signal_color}; background-color: rgba(255,255,255,0.05); padding: 20px; border-radius: 5px; margin-bottom: 20px; border: 1px solid #374151;'>
            <h3 style='margin-top:0px; color: {signal_color}; font-size: 24px;'>{signal_text}</h3>
            <p style='margin-bottom:0px; font-size: 15px; color: #D1D5DB;'><strong>[전략 근거]</strong> {reasoning}</p>
        </div>
        """, unsafe_allow_html=True)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("통계적 상승 승률", f"{win_rate:.1f}%")
        c2.metric("앙상블 평균 수익률", f"{avg_return:.2f}%")
        c3.metric("최대 상승 (Max)", f"{max_ret:.1f}%")
        c4.metric("최대 하락 (Min)", f"{min_ret:.1f}%")

        st.markdown("#### 📌 정량적 매매 액션 플랜")
        
        alloc_level, alloc_ratio = "", ""
        if avg_dist < 15.0: alloc_level, alloc_ratio = "공격적", "70% 이상"
        elif 15.0 <= avg_dist < 25.0: alloc_level, alloc_ratio = "중립적", "50% 전후"
        else: alloc_level, alloc_ratio = "보수적", "30% 이하"
            
        ta_text = ""
        if tech_data:
            p_price = tech_data['price']
            support_line = tech_data['sma20'] if p_price > tech_data['sma20'] else tech_data['sma50']
            rsi_stat = "과매도(매수 유리)" if tech_data['rsi'] < 40 else "과매수(조정 주의)" if tech_data['rsi'] > 60 else "중립"
            vol_stat = "기관 수급 유입" if tech_data['vol_ratio'] > 150 else "평이한 거래량"
            macro_trend = df['^GSPC'].iloc[-1] - df['^GSPC'].iloc[-5]
            macro_stat = "상승 추세" if macro_trend > 0 else "하락 추세"
            
            opt_text = ""
            if tech_data.get('opt_data'):
                od = tech_data['opt_data']
                # [수정 완료] 옵션 관련 핵심 수치에 하이라이트 색상 적용
                opt_text = (f"<br>▶ <b style='color:#F9FAFB;'>스마트머니 옵션(OI) 현황</b> (최근월물: <span class='highlight-num'>{od['expiry']}</span>)<br>"
                            f"&nbsp;&nbsp;&nbsp;&nbsp;• <b>콜옵션(상승) 최대 밀집:</b> 행사가 <b class='highlight-num'>${od['call_strike']:.2f}</b> (미결제약정 <span class='highlight-num'>{od['call_oi']:,}</span>건 / 거래량 <span class='highlight-num'>{od['call_vol']:,}</span>건)<br>"
                            f"&nbsp;&nbsp;&nbsp;&nbsp;• <b>풋옵션(하락) 최대 밀집:</b> 행사가 <b class='highlight-num'>${od['put_strike']:.2f}</b> (미결제약정 <span class='highlight-num'>{od['put_oi']:,}</span>건 / 거래량 <span class='highlight-num'>{od['put_vol']:,}</span>건)")
            else:
                opt_text = "<br>▶ <b>옵션 데이터:</b> 해당 종목의 파생상품 데이터가 존재하지 않거나 제공되지 않음."

            # [수정 완료] 기술적 타점 핵심 수치(가격, RSI, 밀집도 등)에 하이라이트 색상 적용
            ta_text = (f"• <b>현재가:</b> <span class='highlight-num'>{p_price:.2f}</span><br>"
                       f"• <b>이동평균 지지선:</b> <span class='highlight-num'>{support_line:.2f}</span> 부근<br>"
                       f"• <b>RSI (14일):</b> <span class='highlight-num'>{tech_data['rsi']:.1f}</span> ({rsi_stat})<br>"
                       f"• <b>매물대 (VP):</b> 최대 밀집 방어선 <span class='highlight-num'>{tech_data['vp_support']:.2f}</span><br>"
                       f"• <b>거래량 폭발도:</b> 20일 평균 대비 <span class='highlight-num'>{tech_data['vol_ratio']:.1f}%</span> ({vol_stat})<br>"
                       f"• <b>거시(S&P500) 선물 추세:</b> 최근 5일 <span class='highlight-num'>{macro_stat}</span>"
                       f"{opt_text}")
        else:
            ta_text = "기술적 지표 데이터를 불러올 수 없습니다."

        curr_series = df[target_stock].iloc[-window:].values
        curr_norm = (curr_series - np.min(curr_series)) / (np.max(curr_series) - np.min(curr_series))
        curr_slope = curr_norm[-1] - curr_norm[-5] 
        avg_past_slope = np.mean(past_slopes)      
        exit_signal, exit_color, exit_reason = "", "", ""
        slope_diff = curr_slope - avg_past_slope
        
        # [수정 완료] 궤적 기울기 값(모멘텀)에도 하이라이트 적용하여 시인성 확보
        if slope_diff >= -0.05:
            exit_signal, exit_color = "매수 유지", "#EF4444"
            exit_reason = f"현재 최근 5일 궤적(<span class='highlight-num'>{curr_slope:+.3f}</span>)이 과거 평균(<span class='highlight-num'>{avg_past_slope:+.3f}</span>)을 정상 추종 중입니다."
        elif -0.15 <= slope_diff < -0.05:
            exit_signal, exit_color = "관망 (경고)", "#9CA3AF"
            exit_reason = f"상승 탄력이 과거 평균보다 둔화되며 하단 이탈 중입니다. 지지선 붕괴를 주의하십시오."
        else:
            exit_signal, exit_color = "매도 (이탈 확정)", "#3B82F6"
            exit_reason = f"모멘텀(<span class='highlight-num'>{curr_slope:+.3f}</span>)이 과거 궤적(<span class='highlight-num'>{avg_past_slope:+.3f}</span>)을 하향 이탈했습니다. 시나리오 무효화, 즉각 청산하십시오."

        # 메트릭 카드 텍스트 렌더링 (거리 점수 수치에도 하이라이트 적용)
        st.markdown(f"""
        <div class="metric-card">
            <b style='font-size:16px; color:#F9FAFB;'>① 자금 투입 비중 (DTW 거리 기반):</b> <span style='color:#EF4444; font-size:16px;'>{alloc_level} 진입 ({alloc_ratio})</span><br>
            <span style='color:#9CA3AF; font-size:14px;'>└ 근거: 산출된 평균 유사도 거리 점수는 <span class='highlight-num'>{avg_dist:.2f}</span> (15 미만 공격적, 25 이상 보수적 통제)</span>
        </div>
        <div class="metric-card">
            <b style='font-size:16px; color:#F9FAFB;'>② 진입 타점 정밀화 (기술적/파생 지표):</b><br>
            <span style='font-size:14.5px; line-height: 1.6; color:#D1D5DB;'>{ta_text}</span>
        </div>
        <div class="metric-card">
            <b style='font-size:16px; color:#F9FAFB;'>③ 궤적 추적 청산 전략:</b> <span style='color:{exit_color}; font-size:16px;'>{exit_signal}</span><br>
            <span style='color:#9CA3AF; font-size:14px;'>└ 근거: {exit_reason}</span>
        </div>
        """, unsafe_allow_html=True)


        st.markdown('<div class="section-header">📈 2. 궤적 추적 시뮬레이션 차트</div>', unsafe_allow_html=True)
        path_fig = go.Figure()
        
        path_fig.add_trace(go.Scatter(y=curr_norm, mode='lines', name='현재 실제 경로', line=dict(color='#EF4444', width=4)))
        
        for rank, (match_idx, _) in enumerate(top_matches):
            past_full = df[target_stock].iloc[match_idx : match_idx + window + 20].values
            past_norm = (past_full - np.min(past_full[:window])) / (np.max(past_full[:window]) - np.min(past_full[:window]))
            path_fig.add_trace(go.Scatter(y=past_norm, mode='lines', name=f'과거 {rank+1}위 시나리오', line=dict(width=2, color=f'rgba(59, 130, 246, {1.0 - rank*0.15})')))
            
        # [수정 완료] 점선 및 텍스트 상자 다크 모드 색상 튜닝
        path_fig.add_shape(
            type="line", x0=window-1, x1=window-1, 
            y0=0, y1=1, xref='x', yref='paper', 
            line=dict(color="#9CA3AF", width=2, dash="dot")
        )
        
        path_fig.add_annotation(
            x=window-1, y=1, xref='x', yref='paper', 
            text="현재 시점 (미래 프로젝션 분기점)", 
            showarrow=True, arrowhead=1, ax=50, ay=0,
            font=dict(size=13, color="#F9FAFB"),
            bgcolor="#1F2937", bordercolor="#4B5563", borderpad=4
        )
        
        # [수정 완료] 차트 배경 투명화 및 그리드(눈금선) 톤다운
        path_fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=30, b=10), 
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5, font=dict(color="#D1D5DB")), 
            xaxis=dict(title="경과 일수", showgrid=True, gridcolor='#374151', title_font=dict(color="#9CA3AF"), tickfont=dict(color="#9CA3AF")), 
            yaxis=dict(title="정규화 스케일", showgrid=True, gridcolor='#374151', title_font=dict(color="#9CA3AF"), tickfont=dict(color="#9CA3AF"))
        )
        st.plotly_chart(path_fig, use_container_width=True)


        st.markdown('<div class="section-header">🔍 3. 기초 데이터 및 가중치 분석</div>', unsafe_allow_html=True)
        col_w, col_t = st.columns([1, 2])
        
        with col_w:
            st.markdown("**매크로 지표 동적 가중치**")
            weight_fig = go.Figure([go.Bar(x=list(feature_weights.keys()), y=list(feature_weights.values()), marker_color='#3B82F6')])
            # [수정 완료] 가중치 차트 배경 투명화
            weight_fig.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', xaxis=dict(tickfont=dict(color="#9CA3AF")), yaxis=dict(tickfont=dict(color="#9CA3AF")))
            st.plotly_chart(weight_fig, use_container_width=True)
            
        with col_t:
            st.markdown("**클린 상위 매칭 구간 원본 데이터**")
            match_data = []
            for rank, (match_idx, dist_score) in enumerate(top_matches):
                m_start = df.index[match_idx].strftime('%Y-%m-%d')
                m_end = df.index[match_idx + window].strftime('%Y-%m-%d')
                ret = returns_list[rank]
                match_data.append({"순위": f"{rank+1}위", "시작일": m_start, "종료일(현재시점)": m_end, "거리점수(일치도)": round(dist_score, 2), "이후 20일 수익률": f"{ret:+.2f}%"})
            st.dataframe(pd.DataFrame(match_data), use_container_width=True, hide_index=True)
