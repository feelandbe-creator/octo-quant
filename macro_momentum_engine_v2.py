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

st.markdown("""
<style>
    .report-title { font-size: 22px; font-weight: 800; color: #F9FAFB; margin-bottom: 0px; }
    .report-subtitle { font-size: 16px; color: #9CA3AF; margin-bottom: 30px; border-bottom: 2px solid #374151; padding-bottom: 10px; }
    .section-header { font-size: 22px; font-weight: 700; color: #F9FAFB; margin-top: 40px; margin-bottom: 15px; border-left: 4px solid #3B82F6; padding-left: 10px; }
    .metric-card { background-color: rgba(255,255,255,0.03); padding: 15px; border-radius: 8px; border: 1px solid #374151; margin-bottom: 15px; color: #E5E7EB; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="report-title">🛡️ AI 퀀트 터미널 : V4.9 (옥토만경)</div>', unsafe_allow_html=True)
st.markdown('<div class="report-subtitle">실시간 정량적 실전 매매 지침 및 파생(OI) 수급 추적 엔진 탑재 리포트</div>', unsafe_allow_html=True)

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
        
        # 메인 시그널
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
            signal_text, signal_color = f"적극 매도 (향후 20일 {abs(avg_return):.2f}% 하락 예상)", "#3B82F6"
            reasoning = "강력한 하방 압력 및 바닥권 승률이 겹친 구간이므로 즉각적인 포지션 정리가 필요합니다."

        st.markdown(f"""
        <div style='border-left: 5px solid {signal_color}; background-color: rgba(255,255,255,0.05); padding: 20px; border-radius: 5px; margin-bottom: 20px; border: 1px solid #374151;'>
            <h3 style='margin-top:0px; color: {signal_color}; font-size: 24px;'>{signal_text}</h3>
            <p style='margin-bottom:0px; font-size: 15px; color: #D1D5DB;'><strong>[전략 근거]</strong> {reasoning}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 2x2 그리드 단추
        win_rate_color = "#EF4444" if win_rate >= 50 else "#3B82F6"
        avg_ret_color = "#EF4444" if avg_return > 0 else "#3B82F6"
        max_ret_color = "#EF4444"
        min_ret_color = "#3B82F6"
        
        st.markdown(f"""
        <div style="position: relative; margin-bottom: 20px;">
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                <div style="text-align: center; background-color: rgba(255,255,255,0.03); padding: 20px; border-radius: 8px; border: 1px solid #374151;">
                    <div style="color: #9CA3AF; font-size: 14px; margin-bottom: 5px;">통계적 상승 승률</div>
                    <div style="color: {win_rate_color}; font-size: 28px; font-weight: 700;">{win_rate:.1f}%</div>
                </div>
                <div style="text-align: center; background-color: rgba(255,255,255,0.03); padding: 20px; border-radius: 8px; border: 1px solid #374151;">
                    <div style="color: #9CA3AF; font-size: 14px; margin-bottom: 5px;">기간 평균 수익률</div>
                    <div style="color: {avg_ret_color}; font-size: 28px; font-weight: 700;">{avg_return:+.2f}%</div>
                </div>
                <div style="text-align: center; background-color: rgba(255,255,255,0.03); padding: 20px; border-radius: 8px; border: 1px solid #374151;">
                    <div style="color: #9CA3AF; font-size: 14px; margin-bottom: 5px;">최대 상승 (Max)</div>
                    <div style="color: {max_ret_color}; font-size: 28px; font-weight: 700;">{max_ret:+.1f}%</div>
                </div>
                <div style="text-align: center; background-color: rgba(255,255,255,0.03); padding: 20px; border-radius: 8px; border: 1px solid #374151;">
                    <div style="color: #9CA3AF; font-size: 14px; margin-bottom: 5px;">최대 하락 (Min)</div>
                    <div style="color: {min_ret_color}; font-size: 28px; font-weight: 700;">{min_ret:+.1f}%</div>
                </div>
            </div>
            <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); 
                        width: 70px; height: 70px; background-color: #1F2937; border: 3px solid #374151; 
                        border-radius: 50%; display: flex; align-items: center; justify-content: center; 
                        color: #F9FAFB; font-weight: 800; font-size: 16px; box-shadow: 0 4px 10px rgba(0,0,0,0.5); 
                        z-index: 10;">
                20일
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 액션 플랜 데이터 연산
        def fmt_trend_color(val):
            return f"<span style='color:#EF4444; font-weight:700;'>{val:+.3f}</span>" if val > 0 else f"<span style='color:#3B82F6; font-weight:700;'>{val:+.3f}</span>"

        alloc_level, alloc_ratio = "", ""
        if avg_dist < 15.0: alloc_level, alloc_ratio = "공격적", "70% 이상"
        elif 15.0 <= avg_dist < 25.0: alloc_level, alloc_ratio = "중립적", "50% 전후"
        else: alloc_level, alloc_ratio = "보수적", "30% 이하"
            
        ta_text = ""
        if tech_data:
            p_price = tech_data['price']
            
            if p_price >= tech_data['sma20']:
                ma_val = tech_data['sma20']
                ma_label = "20일선 지지"
                ma_color = "#EF4444"
            else:
                ma_val = tech_data['sma20']
                ma_label = "20일선 저항"
                ma_color = "#3B82F6"
                
            if tech_data['rsi'] < 40:
                rsi_stat = "과매도(<span style='color:#EF4444; font-weight:700;'>매수 유리</span>)"
            elif tech_data['rsi'] > 60:
                rsi_stat = "과매수(<span style='color:#3B82F6; font-weight:700;'>매도 유리</span>)"
            else:
                rsi_stat = "중립"
            
            vol_color = "#EF4444" if tech_data['vol_ratio'] >= 100 else "#3B82F6"
            vol_stat = "상승 지표" if tech_data['vol_ratio'] >= 100 else "하락 지표"
            
            macro_trend = df['^GSPC'].iloc[-1] - df['^GSPC'].iloc[-5]
            macro_stat = "<span style='color:#EF4444; font-weight:700;'>상승 추세</span>" if macro_trend > 0 else "<span style='color:#3B82F6; font-weight:700;'>하락 추세</span>"
            
            vp_val = tech_data['vp_support']
            if p_price >= vp_val:
                vp_label = "지지선"
                vp_color = "#EF4444"
            else:
                vp_label = "저항선"
                vp_color = "#3B82F6"
            
            opt_text = ""
            if tech_data.get('opt_data'):
                od = tech_data['opt_data']
                opt_text = (f"<br>▶ <b style='color:#F9FAFB;'>스마트머니 옵션(OI) 현황</b> (최근월물: <b>{od['expiry']}</b>)<br>"
                            f"&nbsp;&nbsp;&nbsp;&nbsp;• <b>콜옵션(상승) 최대 밀집:</b> 행사가 <b style='color:#EF4444;'>${od['call_strike']:.2f}</b> (미결제약정 <b style='color:#EF4444;'>{od['call_oi']:,}</b>건 / 거래량 <b style='color:#EF4444;'>{od['call_vol']:,}</b>건)<br>"
                            f"&nbsp;&nbsp;&nbsp;&nbsp;• <b>풋옵션(하락) 최대 밀집:</b> 행사가 <b style='color:#3B82F6;'>${od['put_strike']:.2f}</b> (미결제약정 <b style='color:#3B82F6;'>{od['put_oi']:,}</b>건 / 거래량 <b style='color:#3B82F6;'>{od['put_vol']:,}</b>건)")
            else:
                opt_text = "<br>▶ <b>옵션 데이터:</b> 해당 종목의 파생상품 데이터가 존재하지 않거나 제공되지 않음."

            ta_text = (f"• <b>현재가:</b> <b>{p_price:.2f}</b><br>"
                       f"• <b>이동평균 타점 ({ma_label}):</b> <b style='color:{ma_color};'>{ma_val:.2f}</b> 부근<br>"
                       f"• <b>RSI (14일):</b> <b>{tech_data['rsi']:.1f}</b> {rsi_stat}<br>"
                       f"• <b>매물대 (VP) 최대 밀집 {vp_label}:</b> <b style='color:{vp_color};'>{vp_val:.2f}</b><br>"
                       f"• <b>거래량 폭발도:</b> 20일 평균 대비 <b style='color:{vol_color};'>{tech_data['vol_ratio']:.1f}%</b> ({vol_stat})<br>"
                       f"• <b>거시(S&P500) 선물 추세:</b> 최근 5일 {macro_stat}"
                       f"{opt_text}")
        else:
            ta_text = "기술적 지표 데이터를 불러올 수 없습니다."

        curr_slope = curr_norm[-1] - curr_norm[-5] 
        avg_past_slope = np.mean(past_slopes)      
        exit_signal, exit_color, exit_reason = "", "", ""
        slope_diff = curr_slope - avg_past_slope
        
        curr_slope_str = fmt_trend_color(curr_slope)
        avg_past_slope_str = fmt_trend_color(avg_past_slope)
        
        if slope_diff >= 0.05:
            exit_signal, exit_color = "공격적 매수", "#EF4444" 
            exit_reason = f"상승 탄력({curr_slope_str})이 과거 평균({avg_past_slope_str})을 강하게 상회하며 폭발 중입니다. 저항선 돌파가 확인되면 추격 매수를 통해 단기 수익을 극대화하는 <b>'공격적 매수'</b> 전술이 유효합니다."
        elif -0.05 <= slope_diff < 0.05:
            exit_signal, exit_color = "점진적 매수", "#EF4444" 
            exit_reason = f"현재 궤적({curr_slope_str})이 과거 평균({avg_past_slope_str})을 안정적으로 추종하고 있습니다. 주요 지지선 부근에서 단기 조정이 올 때마다 물량을 모아가는 <b>'점진적 매수'</b> 전술을 권장합니다."
        elif -0.15 <= slope_diff < -0.05:
            exit_signal, exit_color = "관망 (점진적 매도)", "#9CA3AF" 
            exit_reason = f"상승 탄력이 과거 궤적보다 둔화되며 하단 이탈 중입니다. 지지선 붕괴를 주의하십시오. 아직 탈출 기회가 남아있을 수 있으니, 장중 반등이 올 때마다 비중을 축소하면서 <b>'질서 있는 후퇴(점진적 매도)'</b> 전술을 취해야 합니다."
        else:
            exit_signal, exit_color = "공격적 매도", "#3B82F6" 
            exit_reason = f"모멘텀({curr_slope_str})이 과거 궤적({avg_past_slope_str})을 하향 이탈 확정했습니다. 과거의 상승 시나리오가 완전히 무효화되었으므로, 즉각적인 투매로 하방 리스크를 차단하는 <b>'공격적 매도'</b> 전술을 집행하십시오."

        st.markdown("#### 📌 정량적 매매 액션 플랜")
        st.markdown(f"""
        <div class="metric-card">
            <b style='font-size:16px; color:#F9FAFB;'>① 자금 투입 비중 (DTW 거리 기반):</b> <span style='color:#EF4444; font-size:16px;'>{alloc_level} 진입 ({alloc_ratio})</span><br>
            <span style='color:#9CA3AF; font-size:14px;'>└ 근거: 산출된 평균 유사도 거리 점수는 <b>{avg_dist:.2f}</b> (15 미만 공격적, 25 이상 보수적 통제)</span>
        </div>
        <div class="metric-card">
            <b style='font-size:16px; color:#F9FAFB;'>② 진입 타점 정밀화 (기술적/파생 지표):</b><br>
            <span style='font-size:14.5px; line-height: 1.6; color:#D1D5DB;'>{ta_text}</span>
        </div>
        <div class="metric-card">
            <b style='font-size:16px; color:#F9FAFB;'>③ 궤적 추적 단기 전술:</b> <span style='color:{exit_color}; font-size:16px; font-weight:700;'>{exit_signal}</span><br>
            <span style='color:#9CA3AF; font-size:14.5px; line-height: 1.6;'>└ <b>[실전 가이드]</b> {exit_reason}</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-header">🔍 2. 앙상블 패턴 매칭 분석 (과거 상위 국면 상세)</div>', unsafe_allow_html=True)
        
        for i in range(0, len(top_matches), 3):
            chunk = top_matches[i:i+3]
            cols = st.columns(len(chunk))
            for rank, (match_idx, dist_score) in enumerate(chunk):
                actual_rank = i + rank + 1
                m_start = df.index[match_idx].strftime('%Y-%m-%d')
                m_end = df.index[match_idx + window].strftime('%Y-%m-%d')
                ret = returns_list[i + rank]
                ret_color = "#EF4444" if ret > 0 else "#3B82F6"
                
                with cols[rank]:
                    st.markdown(f"""
                    <div style='background-color: rgba(255,255,255,0.03); padding: 15px; border-radius: 8px; border: 1px solid #374151; margin-bottom: 10px;'>
                        <div style='color: #F9FAFB; font-weight: 700; font-size: 15px; margin-bottom: 5px; border-bottom: 1px solid #4B5563; padding-bottom: 5px;'>[과거 상위 {actual_rank}위 패턴]</div>
                        <div style='color: #9CA3AF; font-size: 13.5px;'>📅 {m_start} ~ {m_end}</div>
                        <div style='margin-top: 10px; font-size: 14px; color: #D1D5DB;'>패턴 거리 점수 (일치도): <b>{dist_score:.2f}</b></div>
                        <div style='font-size: 14px; color: #D1D5DB; margin-top: 5px;'>이후 20일 실제 수익률: <b style='color: {ret_color}; font-size: 16px;'>{ret:+.2f}%</b></div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">📈 3. 궤적 추적 시뮬레이션 차트</div>', unsafe_allow_html=True)
        path_fig = go.Figure()
        
        path_fig.add_trace(go.Scatter(y=curr_norm, mode='lines', name='현재 실제 경로', line=dict(color='#EF4444', width=4)))
        
        for rank, (match_idx, _) in enumerate(top_matches):
            past_full = df[target_stock].iloc[match_idx : match_idx + window + 20].values
            past_norm = (past_full - np.min(past_full[:window])) / (np.max(past_full[:window]) - np.min(past_full[:window]))
            path_fig.add_trace(go.Scatter(y=past_norm, mode='lines', name=f'과거 {rank+1}위 시나리오', line=dict(width=2, color=f'rgba(59, 130, 246, {1.0 - rank*0.15})')))
            
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
        
        path_fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=30, b=10), 
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5, font=dict(color="#D1D5DB")), 
            xaxis=dict(title="경과 일수", showgrid=True, gridcolor='#374151', title_font=dict(color="#9CA3AF"), tickfont=dict(color="#9CA3AF")), 
            yaxis=dict(title="정규화 스케일", showgrid=True, gridcolor='#374151', title_font=dict(color="#9CA3AF"), tickfont=dict(color="#9CA3AF"))
        )
        st.plotly_chart(path_fig, use_container_width=True)

        st.markdown('<div class="section-header">🔍 4. 거시 지표 가중치 분석</div>', unsafe_allow_html=True)
        st.markdown("**현재 시장을 지배하는 매크로 변수 동적 가중치**")
        weight_fig = go.Figure([go.Bar(x=list(feature_weights.keys()), y=list(feature_weights.values()), marker_color='#3B82F6')])
        weight_fig.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', xaxis=dict(tickfont=dict(color="#9CA3AF")), yaxis=dict(tickfont=dict(color="#9CA3AF")))
        st.plotly_chart(weight_fig, use_container_width=True)
