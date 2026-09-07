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

# =============================================================================
# 수정 요약 (V4.9 -> V5.0)
# 1. [치명적] 상관관계 계산을 '가격 레벨' -> '수익률(pct_change)' 기준으로 변경
#    (비정상 시계열 간 허위상관 문제 해결)
# 2. [치명적] 표본 수가 작을 때(특히 top_n < 5, 또는 클린 매칭 부족) 명시적
#    경고 문구 추가 - "100% 승률" 같은 표현이 과신을 유발하지 않도록
# 3. [치명적] VIX/이벤트 배제 필터가 통계를 낙관적으로 편향시킨다는 경고 추가
# 4. [치명적] DTW 거리값을 (윈도우 길이 x 유효 피처 수)로 정규화하여
#    종목/구간마다 스케일이 달라지는 문제 완화, 임계값도 재조정
# 5. historical_data 길이가 음수/부족한 경우 크래시 대신 안내 메시지
# 6. 수익률 계산 시 0/NaN 나눗셈 가드
# 7. datetime.utcnow() deprecated -> datetime.now(timezone.utc)
# 8. 뉴욕 시간 계산에 서머타임(EDT/EST) 반영
# 9. 티커 fetch 실패 시 어떤 티커가 실패했는지 사용자에게 표시
# 10. st.cache_data에 ttl 추가 (30분)
# 11. RSI를 Wilder 방식(지수이동평균)으로 변경
# 12. 옵션 OI 해석에 캐비어트 문구 추가
#
# --- 추가 반영 (V5.0 -> V6.0) ---
# 13. [설계개선] DTW 매칭 피처에 종목 자신의 가격궤적을 포함
#     (기존엔 매크로 지표만으로 국면을 매칭하고, 수익률만 종목 자체 것을 사용
#      → "매크로 환경이 비슷하면 종목도 비슷하게 움직인다"는 검증되지 않은
#      가정에 의존. build_weighted_features()에서 종목 자체 궤적에
#      target_weight_share(기본 50%, 슬라이더로 조절) 비중을 고정 배분)
# 14. [설계개선] 워크포워드 검증 모듈 신규 추가 (run_walkforward_validation)
#     - 과거 여러 시점에서 "그 시점까지의 데이터만으로" 동일 로직을 재현해
#       신호를 뽑고, 실제 이후 20일 수익률과 방향이 맞았는지 누적 검증
#     - 적중률(Hit Rate), 정보계수(IC), 매수/매도 신호그룹별 평균 실제수익률을
#       제공해 "이 전략이 과거에 진짜 예측력이 있었는가"를 별도로 확인 가능
#     - 계산 비용 때문에 별도 버튼으로 분리, 탐색 풀을 최근 N거래일로 캡핑한
#       근사 검증임을 명시
# =============================================================================

# --- 0. 중대 역사적 이벤트 사전 ---
MAJOR_EVENTS = {
    "2026-02-28": "미국-이란 전쟁 시작",
    "2022-02-24": "러시아-우크라이나 전쟁 발발",
    "2020-03-11": "WHO 코로나19 팬데믹 선언",
    "2020-03-23": "연준 무제한 양적완화(QE) 발표",
    "2018-10-03": "미국 10년물 국채금리 7년 최고치 돌파",
    "2015-08-24": "중국 위안화 쇼크 (블랙 먼데이)"
}

MIN_RELIABLE_SAMPLE = 5  # 이보다 표본이 적으면 통계 신뢰도 경고


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


@st.cache_data(show_spinner=False, ttl=1800)
def fetch_comprehensive_market_data(tickers, start_date, end_date):
    compiled_data = pd.DataFrame()
    failed_tickers = []
    for ticker in tickers:
        try:
            raw_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if raw_data.empty:
                failed_tickers.append(ticker)
                continue
            if 'Adj Close' in raw_data.columns:
                price_series = raw_data['Adj Close'].squeeze()
            elif 'Close' in raw_data.columns:
                price_series = raw_data['Close'].squeeze()
            else:
                failed_tickers.append(ticker)
                continue
            compiled_data[ticker] = price_series
        except Exception:
            failed_tickers.append(ticker)
            continue
    compiled_data.ffill(inplace=True)
    compiled_data.dropna(inplace=True)
    return compiled_data, failed_tickers


@st.cache_data(show_spinner=False, ttl=1800)
def fetch_technical_indicators(ticker):
    try:
        end = datetime.date.today()
        start = end - datetime.timedelta(days=100)
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            return None

        close = df['Close'].squeeze() if 'Close' in df.columns else df['Adj Close'].squeeze()
        vol = df['Volume'].squeeze() if 'Volume' in df.columns else pd.Series(0, index=close.index)

        sma20 = float(close.rolling(window=20).mean().iloc[-1])
        sma50 = float(close.rolling(window=50).mean().iloc[-1])

        # --- RSI: Wilder 방식(지수이동평균)으로 계산 ---
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi_series = 100 - (100 / (1 + rs))
        rsi14 = float(rsi_series.iloc[-1]) if not np.isnan(rsi_series.iloc[-1]) else 50.0

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


def build_weighted_features(df, macro_tickers, target_stock, window_size, target_weight_share=0.5):
    """
    [신규] DTW 매칭에 사용할 피처 행렬을 만든다.
    기존에는 매크로 티커들만 매칭 대상이었으나, 이제 종목 자신의 가격궤적도
    하나의 피처로 포함시킨다 (target_weight_share 비중만큼 고정 배분).
    나머지 (1 - target_weight_share) 비중은 매크로 티커들이 수익률 상관관계
    크기에 비례해 나눠 갖는다.

    반환: (weighted_scaled_features, weights_dict, feature_order) 또는
          데이터 부족 시 (None, {}, [])
    """
    if target_stock not in df.columns:
        return None, {}, []
    valid_macros = [t for t in macro_tickers if t in df.columns]
    if not valid_macros:
        return None, {}, []

    returns_df = df.pct_change().dropna()
    if len(returns_df) < window_size:
        return None, {}, []

    recent_window_returns = returns_df.iloc[-window_size:]
    correlations = {}
    for ticker in valid_macros:
        corr = recent_window_returns[target_stock].corr(recent_window_returns[ticker])
        correlations[ticker] = abs(corr) if not np.isnan(corr) else 0.0

    total_corr = sum(correlations.values())
    if total_corr > 0:
        macro_weights_norm = {k: v / total_corr for k, v in correlations.items()}
    else:
        macro_weights_norm = {k: 1.0 / len(valid_macros) for k in valid_macros}

    target_weight_share = float(np.clip(target_weight_share, 0.0, 0.95))
    remaining = 1.0 - target_weight_share
    weights = {target_stock: target_weight_share}
    for k, v in macro_weights_norm.items():
        weights[k] = v * remaining

    feature_order = valid_macros + [target_stock]
    feature_data = df[feature_order]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_data)
    weight_vector = np.array([np.sqrt(weights[t]) for t in feature_order])
    weighted_scaled = scaled * weight_vector

    return weighted_scaled, weights, feature_order


def find_top_historical_matches(df, macro_tickers, target_stock, window_size, top_n=5, target_weight_share=0.5):
    """
    반환값에 다음이 추가됨:
    - normalized avg distance (피처수 x 윈도우 로 정규화)
    - 표본 수 관련 경고 플래그
    """
    if target_stock not in df.columns:
        return [], {}, [], {"error": "target_missing"}

    # --- [수정] 매크로 + 종목 자신의 가격궤적을 함께 DTW 피처로 사용 ---
    weighted_scaled_macro, weights, feature_order = build_weighted_features(
        df, macro_tickers, target_stock, window_size, target_weight_share
    )
    if weighted_scaled_macro is None:
        return [], {}, [], {"error": "insufficient_history"}

    current_pattern = weighted_scaled_macro[-window_size:]

    # --- [수정 5] 데이터 부족 시 크래시 방지 ---
    cutoff = window_size + 20
    if len(weighted_scaled_macro) <= cutoff + window_size:
        return [], weights, [], {"error": "insufficient_history"}
    historical_data = weighted_scaled_macro[:-cutoff]

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

        past_window = historical_data[i: i + window_size]
        distance, _ = fastdtw(current_pattern, past_window, dist=euclidean)

        if event_alerts:
            excluded_distances.append((i, distance, event_alerts))
        else:
            clean_distances.append((i, distance))

    clean_distances.sort(key=lambda x: x[1])
    top_matches = []
    selected_indices = []
    for idx, dist in clean_distances:
        if len(top_matches) >= top_n:
            break
        if any(abs(idx - s_idx) < window_size for s_idx in selected_indices):
            continue
        top_matches.append((idx, dist))
        selected_indices.append(idx)

    excluded_distances.sort(key=lambda x: x[1])
    top_excluded = excluded_distances[:2]

    meta = {
        "error": None,
        "n_clean_total": len(clean_distances),
        "n_excluded_total": len(excluded_distances),
        "n_features": len(feature_order),
    }

    return top_matches, weights, top_excluded, meta


# =============================================================================
# --- 4. 워크포워드 검증 모듈 (신규) ---
# "이 매칭 로직이 과거에 실제로 방향성을 맞췄는가"를 검증한다.
# 매 테스트 시점 T마다 T 이전 데이터만으로 동일한 매칭을 재현하여 신호를
# 뽑고, 실제 T+20일 수익률과 방향이 맞았는지를 누적 집계한다.
# 속도를 위해 탐색 풀을 최근 pool_cap 거래일로 캡핑하고 VIX/이벤트 필터는
# 생략한다 — 따라서 실거래 로직과 100% 동일하지 않은 "근사 검증"이다.
# =============================================================================
def run_walkforward_validation(df, macro_tickers, target_stock, window_size, top_n,
                                target_weight_share=0.5, n_test_points=15, pool_cap=800):
    total_len = len(df)
    earliest_test_idx = max(window_size * 4, 100)
    latest_test_idx = total_len - 20 - 1  # 실제 미래 수익률(T+20)이 존재해야 함

    if latest_test_idx <= earliest_test_idx:
        return None

    candidate_range = latest_test_idx - earliest_test_idx + 1
    n_points = min(n_test_points, candidate_range)
    test_indices = sorted(set(
        np.linspace(earliest_test_idx, latest_test_idx, num=n_points, dtype=int).tolist()
    ))

    results = []
    for t in test_indices:
        pool_start = max(0, t - pool_cap)
        df_pool = df.iloc[pool_start:t + 1]  # T 시점까지의 데이터만 사용 (미래 누설 없음)

        weighted_features, _, feature_order = build_weighted_features(
            df_pool, macro_tickers, target_stock, window_size, target_weight_share
        )
        if weighted_features is None or len(weighted_features) < window_size * 3:
            continue

        current_pattern = weighted_features[-window_size:]
        cutoff = window_size + 20
        if len(weighted_features) <= cutoff + window_size:
            continue
        historical_pool = weighted_features[:-cutoff]

        distances = []
        for i in range(len(historical_pool) - window_size):
            past_window = historical_pool[i:i + window_size]
            dist, _ = fastdtw(current_pattern, past_window, dist=euclidean)
            distances.append((i, dist))
        distances.sort(key=lambda x: x[1])

        matches, selected = [], []
        for idx, dist in distances:
            if len(matches) >= top_n:
                break
            if any(abs(idx - s) < window_size for s in selected):
                continue
            matches.append((idx, dist))
            selected.append(idx)
        if not matches:
            continue

        match_returns = []
        for idx, _ in matches:
            p_c = df_pool[target_stock].iloc[idx + window_size]
            p_f = df_pool[target_stock].iloc[idx + window_size + 20]
            if p_c and not np.isnan(p_c) and p_c != 0:
                match_returns.append(((p_f - p_c) / p_c) * 100)
        if not match_returns:
            continue

        predicted_avg = float(np.mean(match_returns))
        predicted_dir = 1 if predicted_avg > 0 else -1

        p_now = df[target_stock].iloc[t]
        p_future = df[target_stock].iloc[t + 20]
        if not p_now or np.isnan(p_now) or p_now == 0:
            continue
        actual_return = float(((p_future - p_now) / p_now) * 100)
        actual_dir = 1 if actual_return > 0 else -1

        results.append({
            "date": df.index[t],
            "predicted_return": predicted_avg,
            "actual_return": actual_return,
            "hit": predicted_dir == actual_dir,
        })

    return results


# --- 3. UI/UX 대시보드 ---
st.set_page_config(page_title="AI 퀀트 터미널(옥토만경)", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .report-title { font-size: 22px; font-weight: 800; color: #F9FAFB; margin-bottom: 0px; }
    .report-subtitle { font-size: 16px; color: #9CA3AF; margin-bottom: 30px; border-bottom: 2px solid #374151; padding-bottom: 10px; }
    .section-header { font-size: 22px; font-weight: 700; color: #F9FAFB; margin-top: 40px; margin-bottom: 15px; border-left: 4px solid #3B82F6; padding-left: 10px; }
    .metric-card { background-color: rgba(255,255,255,0.03); padding: 15px; border-radius: 8px; border: 1px solid #374151; margin-bottom: 15px; color: #E5E7EB; }
    .caveat-box { background-color: rgba(251,191,36,0.08); border: 1px solid #92400E; padding: 12px 15px; border-radius: 6px; color: #FCD34D; font-size: 13.5px; margin-bottom: 15px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="report-title">🛡️ AI 퀀트 터미널 : V6.0 (옥토만경)</div>', unsafe_allow_html=True)
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

    target_weight_share = st.slider(
        "매칭 시 '종목 자신의 궤적' 가중치 비중",
        min_value=0.2, max_value=0.8, value=0.5, step=0.1,
        help="높일수록 '내 종목의 과거 유사 패턴'을 우선 찾고, 낮출수록 '비슷한 거시환경이었던 국면'을 우선 찾습니다. "
             "나머지 비중은 매크로 티커들이 최근 수익률 상관관계 크기에 비례해 나눠 가집니다."
    )

    run_sim = st.button("⚙️ 시뮬레이션 시작", use_container_width=True, type="primary")

if run_sim:
    with st.spinner(f"'{raw_input}' 정밀 데이터 스크래핑 및 인공지능 매매 판독 중..."):

        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=365 * lookback_years)
        macro_tickers = ['QQQ', '^GSPC', 'DIA', '^TNX', 'DX=F', '^VIX', 'CL=F']
        all_tickers = list(set(macro_tickers + [target_stock]))

        df, failed_tickers = fetch_comprehensive_market_data(
            all_tickers, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
        )
        tech_data = fetch_technical_indicators(target_stock)

        if target_stock not in df.columns:
            st.error(f"⚠️ '{raw_input}' 데이터를 수신하지 못했습니다. 상장 폐지되었거나 티커가 올바르지 않습니다.")
            st.stop()

        if failed_tickers:
            st.caption(f"⚠️ 데이터 수신 실패 티커 (분석에서 제외됨): {', '.join(failed_tickers)}")

        top_matches, feature_weights, top_excluded, meta = find_top_historical_matches(
            df, macro_tickers, target_stock, window_size=window, top_n=top_n_input,
            target_weight_share=target_weight_share
        )

        if meta.get("error") == "insufficient_history":
            st.error("⚠️ 선택하신 윈도우/탐색 깊이 조합으로는 비교할 과거 데이터가 부족합니다. 탐색 깊이를 늘리거나 윈도우를 줄여주세요.")
            st.stop()

        if top_excluded:
            st.warning("⚠️ **[시스템 알림] 대외 돌발 변수 격리 조치 완료**")
            for idx, dist, alerts in top_excluded:
                ex_start = df.index[idx].strftime('%Y-%m-%d')
                ex_end = df.index[idx + window].strftime('%Y-%m-%d')
                event_str = ", ".join(alerts)
                st.caption(f" └ 과거 구간 [{ex_start} ~ {ex_end}] 내에 **{event_str}**이 포함되어 있어 통계에서 원천 배제되었습니다.")
            st.markdown(
                '<div class="caveat-box">⚠️ <b>주의:</b> 이 필터는 VIX 급등 구간과 주요 위기 이벤트가 겹친 구간을 백테스트에서 제외합니다. '
                '따라서 아래 승률·기대수익률은 상대적으로 "평온했던 시장"만을 기준으로 산출된 것이며, '
                '실제 하방 리스크(급락 국면)는 과소평가되어 있을 수 있습니다.</div>',
                unsafe_allow_html=True
            )

        if not top_matches:
            st.error("⚠️ 클린 데이터가 부족합니다. 탐색 깊이를 늘려주십시오.")
            st.stop()

        # --- 표본 수 경고 ---
        n_samples = len(top_matches)
        if n_samples < MIN_RELIABLE_SAMPLE:
            st.markdown(
                f'<div class="caveat-box">📊 <b>표본 수 주의:</b> 현재 매칭된 과거 유사 국면은 <b>{n_samples}개</b>뿐입니다. '
                f'이 정도 표본으로 산출된 승률·평균수익률은 통계적 신뢰구간이 매우 넓어 "우연"일 가능성을 배제할 수 없습니다. '
                f'참고 지표로만 활용하시고, 단독 매매 근거로 삼지 마십시오.</div>',
                unsafe_allow_html=True
            )

        # --- [수정 8] 서머타임 반영 뉴욕 시간 ---
        try:
            from zoneinfo import ZoneInfo
            utc_now = datetime.datetime.now(datetime.timezone.utc)
            kst_now = utc_now.astimezone(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")
            edt_now = utc_now.astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            utc_now = datetime.datetime.now(datetime.timezone.utc)
            kst_now = (utc_now + datetime.timedelta(hours=9)).strftime("%Y-%m-%d %H:%M:%S")
            edt_now = (utc_now - datetime.timedelta(hours=4)).strftime("%Y-%m-%d %H:%M:%S")

        st.markdown('<div class="section-header">📊 1. 정량적 실전 매매 행동 지침 (Real-time Action Plan)</div>', unsafe_allow_html=True)
        st.markdown(f"*(실시간 데이터 기준 시각: **KST** {kst_now} / **뉴욕** {edt_now})*")

        returns_list, distances_list, past_slopes = [], [], []
        for match_idx, dist_score in top_matches:
            p_curr = df[target_stock].iloc[match_idx + window]
            p_future = df[target_stock].iloc[match_idx + window + 20]
            # --- [수정 6] 0/NaN 나눗셈 가드 ---
            if p_curr and not np.isnan(p_curr) and p_curr != 0:
                returns_list.append(((p_future - p_curr) / p_curr) * 100)
            else:
                continue
            distances_list.append(dist_score)

            past_full = df[target_stock].iloc[match_idx: match_idx + window + 20].values
            denom = (np.max(past_full[:window]) - np.min(past_full[:window]))
            if denom == 0:
                past_norm = np.zeros_like(past_full)
            else:
                past_norm = (past_full - np.min(past_full[:window])) / denom
            past_slopes.append(past_norm[window - 1] - past_norm[window - 5])

        if not returns_list:
            st.error("⚠️ 유효한 수익률 데이터를 계산할 수 없습니다.")
            st.stop()

        avg_return = np.mean(returns_list)
        win_rate = sum(1 for r in returns_list if r > 0) / len(returns_list) * 100
        max_ret = max(returns_list)
        min_ret = min(returns_list)
        avg_dist = np.mean(distances_list)

        # --- [수정 4] 거리 정규화: 피처 수 x 윈도우 길이로 스케일 보정 ---
        n_features = meta.get("n_features", len(macro_tickers))
        normalized_avg_dist = avg_dist / np.sqrt(max(window * n_features, 1))

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
        <div style='border-left: 5px solid {signal_color}; background-color: rgba(255,255,255,0.05); padding: 20px; border-radius: 5px; margin-bottom: 10px; border: 1px solid #374151;'>
            <h3 style='margin-top:0px; color: {signal_color}; font-size: 24px;'>{signal_text}</h3>
            <p style='margin-bottom:0px; font-size: 15px; color: #D1D5DB;'><strong>[전략 근거]</strong> {reasoning}</p>
            <p style='margin-top:8px; margin-bottom:0px; font-size: 12.5px; color: #9CA3AF;'>표본 수: {n_samples}개 매칭 기준 (참고용 지표이며 단독 매매 근거로 사용하지 마십시오)</p>
        </div>
        """, unsafe_allow_html=True)

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

        def fmt_trend_color(val):
            return f"<span style='color:#EF4444; font-weight:700;'>{val:+.3f}</span>" if val > 0 else f"<span style='color:#3B82F6; font-weight:700;'>{val:+.3f}</span>"

        # --- [수정 4] 정규화된 거리로 배분 판단 (임계값도 정규화 스케일에 맞게 재조정) ---
        if normalized_avg_dist < 1.2:
            alloc_level, alloc_ratio = "공격적", "70% 이상"
        elif 1.2 <= normalized_avg_dist < 2.0:
            alloc_level, alloc_ratio = "중립적", "50% 전후"
        else:
            alloc_level, alloc_ratio = "보수적", "30% 이하"

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
                opt_text = (f"<br>▶ <b style='color:#F9FAFB;'>옵션(OI) 미결제약정 현황</b> (최근월물: <b>{od['expiry']}</b>)<br>"
                            f"&nbsp;&nbsp;&nbsp;&nbsp;• <b>콜옵션 최대 밀집:</b> 행사가 <b style='color:#EF4444;'>${od['call_strike']:.2f}</b> (미결제약정 <b style='color:#EF4444;'>{od['call_oi']:,}</b>건 / 거래량 <b style='color:#EF4444;'>{od['call_vol']:,}</b>건)<br>"
                            f"&nbsp;&nbsp;&nbsp;&nbsp;• <b>풋옵션 최대 밀집:</b> 행사가 <b style='color:#3B82F6;'>${od['put_strike']:.2f}</b> (미결제약정 <b style='color:#3B82F6;'>{od['put_oi']:,}</b>건 / 거래량 <b style='color:#3B82F6;'>{od['put_vol']:,}</b>건)<br>"
                            f"&nbsp;&nbsp;&nbsp;&nbsp;<span style='color:#9CA3AF; font-size:12.5px;'>⚠️ OI 쏠림은 마켓메이커 헤지 물량일 수도 있어 방향성을 확정하는 지표가 아닙니다. 참고 정보로만 활용하십시오.</span>")
            else:
                opt_text = "<br>▶ <b>옵션 데이터:</b> 해당 종목의 파생상품 데이터가 존재하지 않거나 제공되지 않음."

            ta_text = (f"• <b>현재가:</b> <b>{p_price:.2f}</b><br>"
                       f"• <b>이동평균 타점 ({ma_label}):</b> <b style='color:{ma_color};'>{ma_val:.2f}</b> 부근<br>"
                       f"• <b>RSI (14일, Wilder):</b> <b>{tech_data['rsi']:.1f}</b> {rsi_stat}<br>"
                       f"• <b>매물대 (VP) 최대 밀집 {vp_label}:</b> <b style='color:{vp_color};'>{vp_val:.2f}</b><br>"
                       f"• <b>거래량 폭발도:</b> 20일 평균 대비 <b style='color:{vol_color};'>{tech_data['vol_ratio']:.1f}%</b> ({vol_stat})<br>"
                       f"• <b>거시(S&P500) 선물 추세:</b> 최근 5일 {macro_stat}"
                       f"{opt_text}")
        else:
            ta_text = "기술적 지표 데이터를 불러올 수 없습니다."

        curr_series = df[target_stock].iloc[-window:].values
        curr_denom = (np.max(curr_series) - np.min(curr_series))
        if curr_denom == 0:
            curr_norm = np.zeros_like(curr_series)
        else:
            curr_norm = (curr_series - np.min(curr_series)) / curr_denom

        curr_slope = curr_norm[-1] - curr_norm[-5]
        avg_past_slope = np.mean(past_slopes)
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
            <b style='font-size:16px; color:#F9FAFB;'>① 자금 투입 비중 (정규화 DTW 거리 기반):</b> <span style='color:#EF4444; font-size:16px;'>{alloc_level} 진입 ({alloc_ratio})</span><br>
            <span style='color:#9CA3AF; font-size:14px;'>└ 근거: 정규화 유사도 거리 점수는 <b>{normalized_avg_dist:.3f}</b> (윈도우·피처 수 보정값, 1.2 미만 공격적 / 2.0 이상 보수적)</span>
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
            chunk = top_matches[i:i + 3]
            cols = st.columns(len(chunk))
            for rank, (match_idx, dist_score) in enumerate(chunk):
                actual_rank = i + rank + 1
                m_start = df.index[match_idx].strftime('%Y-%m-%d')
                m_end = df.index[match_idx + window].strftime('%Y-%m-%d')
                ret = returns_list[i + rank] if i + rank < len(returns_list) else 0.0
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
            past_full = df[target_stock].iloc[match_idx: match_idx + window + 20].values
            denom = (np.max(past_full[:window]) - np.min(past_full[:window]))
            past_norm = (past_full - np.min(past_full[:window])) / denom if denom != 0 else np.zeros_like(past_full)
            path_fig.add_trace(go.Scatter(y=past_norm, mode='lines', name=f'과거 {rank + 1}위 시나리오', line=dict(width=2, color=f'rgba(59, 130, 246, {max(1.0 - rank * 0.15, 0.2)})')))

        path_fig.add_shape(
            type="line", x0=window - 1, x1=window - 1,
            y0=0, y1=1, xref='x', yref='paper',
            line=dict(color="#9CA3AF", width=2, dash="dot")
        )

        path_fig.add_annotation(
            x=window - 1, y=1, xref='x', yref='paper',
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

        st.markdown('<div class="section-header">🔍 4. 매칭 피처 가중치 분석 (종목 자체 궤적 + 거시지표)</div>', unsafe_allow_html=True)
        st.markdown(f"**DTW 매칭에 사용된 피처별 가중치** — `{target_stock}` 자신의 궤적에 **{target_weight_share*100:.0f}%**, 나머지를 매크로 티커들이 수익률 상관관계 크기에 비례해 분배")
        weight_labels = [f"★ {target_stock} (자체 궤적)" if k == target_stock else k for k in feature_weights.keys()]
        weight_colors = ["#EF4444" if k == target_stock else "#3B82F6" for k in feature_weights.keys()]
        weight_fig = go.Figure([go.Bar(x=weight_labels, y=list(feature_weights.values()), marker_color=weight_colors)])
        weight_fig.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', xaxis=dict(tickfont=dict(color="#9CA3AF")), yaxis=dict(tickfont=dict(color="#9CA3AF")))
        st.plotly_chart(weight_fig, use_container_width=True)

        # --- 워크포워드 검증에서 재사용할 수 있도록 설정을 세션에 저장 ---
        st.session_state["wf_ready"] = True
        st.session_state["wf_df"] = df
        st.session_state["wf_macro_tickers"] = macro_tickers
        st.session_state["wf_target_stock"] = target_stock
        st.session_state["wf_window"] = window
        st.session_state["wf_top_n"] = top_n_input
        st.session_state["wf_target_weight_share"] = target_weight_share

# =============================================================================
# --- 5. 워크포워드 검증 (전략 자체의 과거 예측력 검증) ---
# 위 리포트와 별개로, "이 매칭 로직이 과거에 실제로 방향을 맞췄는가"를
# 별도 버튼으로 검증합니다. 계산량이 커서 기본 리포트와 분리했습니다.
# =============================================================================
st.markdown('<div class="section-header">🧪 5. 워크포워드 검증 (전략 자체의 과거 예측력 검증)</div>', unsafe_allow_html=True)

if not st.session_state.get("wf_ready"):
    st.info("먼저 위에서 '⚙️ 시뮬레이션 시작'을 1회 실행하면, 그 설정(종목/윈도우/가중치)으로 워크포워드 검증을 돌릴 수 있습니다.")
else:
    st.markdown(
        '<div class="caveat-box">⚠️ <b>이 검증은 근사치입니다.</b> 속도를 위해 매 시점마다 탐색 풀을 최근 800거래일로 제한하고, '
        'VIX·이벤트 배제 필터는 생략했습니다. 위 리포트와 완전히 동일한 로직은 아니지만, '
        '"매크로+자체궤적 DTW 매칭"이라는 핵심 아이디어 자체에 방향성 예측력이 있는지를 대략적으로 확인하는 용도입니다.</div>',
        unsafe_allow_html=True
    )
    wf_col1, wf_col2 = st.columns(2)
    with wf_col1:
        n_test_points = st.selectbox("검증 시점 개수", options=[10, 15, 20, 30, 50, 80], index=1)
        if n_test_points >= 50:
            st.caption(f"⏱️ {n_test_points}개 시점은 계산량이 많아 1~3분 이상 걸릴 수 있습니다.")
    with wf_col2:
        pool_cap = st.selectbox("탐색 풀 크기 (최근 N거래일, 클수록 느려짐)", options=[500, 800, 1200, 1500], index=1)

    run_wf = st.button("🧪 워크포워드 검증 실행", use_container_width=True)

    if run_wf:
        with st.spinner(f"과거 {n_test_points}개 시점에서 매칭 로직을 재현하여 검증 중... (시간이 다소 걸릴 수 있습니다)"):
            wf_results = run_walkforward_validation(
                st.session_state["wf_df"],
                st.session_state["wf_macro_tickers"],
                st.session_state["wf_target_stock"],
                st.session_state["wf_window"],
                st.session_state["wf_top_n"],
                target_weight_share=st.session_state["wf_target_weight_share"],
                n_test_points=n_test_points,
                pool_cap=pool_cap,
            )

        if not wf_results:
            st.error("⚠️ 검증에 필요한 데이터가 부족합니다. 탐색 깊이를 늘리거나 윈도우를 줄여보세요.")
        else:
            n_tests = len(wf_results)
            hit_rate = np.mean([r["hit"] for r in wf_results]) * 100
            predicted = np.array([r["predicted_return"] for r in wf_results])
            actual = np.array([r["actual_return"] for r in wf_results])
            ic = float(np.corrcoef(predicted, actual)[0, 1]) if n_tests >= 3 else float("nan")

            bullish_mask = predicted > 0
            bearish_mask = ~bullish_mask
            bullish_avg = float(np.mean(actual[bullish_mask])) if bullish_mask.any() else float("nan")
            bearish_avg = float(np.mean(actual[bearish_mask])) if bearish_mask.any() else float("nan")

            if n_tests < MIN_RELIABLE_SAMPLE:
                st.markdown(
                    f'<div class="caveat-box">📊 검증 시점이 {n_tests}개뿐이라 아래 수치의 통계적 신뢰도는 낮습니다.</div>',
                    unsafe_allow_html=True
                )

            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
                <div style="text-align: center; background-color: rgba(255,255,255,0.03); padding: 20px; border-radius: 8px; border: 1px solid #374151;">
                    <div style="color: #9CA3AF; font-size: 14px; margin-bottom: 5px;">방향 적중률</div>
                    <div style="color: {'#EF4444' if hit_rate >= 50 else '#3B82F6'}; font-size: 26px; font-weight: 700;">{hit_rate:.1f}%</div>
                    <div style="color: #6B7280; font-size: 12px;">(50%는 동전 던지기 수준)</div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                ic_str = f"{ic:+.3f}" if not np.isnan(ic) else "N/A"
                st.markdown(f"""
                <div style="text-align: center; background-color: rgba(255,255,255,0.03); padding: 20px; border-radius: 8px; border: 1px solid #374151;">
                    <div style="color: #9CA3AF; font-size: 14px; margin-bottom: 5px;">정보계수 (IC)</div>
                    <div style="color: #F9FAFB; font-size: 26px; font-weight: 700;">{ic_str}</div>
                    <div style="color: #6B7280; font-size: 12px;">(0에 가까우면 예측력 없음)</div>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div style="text-align: center; background-color: rgba(255,255,255,0.03); padding: 20px; border-radius: 8px; border: 1px solid #374151;">
                    <div style="color: #9CA3AF; font-size: 14px; margin-bottom: 5px;">검증 시점 수</div>
                    <div style="color: #F9FAFB; font-size: 26px; font-weight: 700;">{n_tests}개</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-card">
                <b style='color:#F9FAFB;'>신호 그룹별 실제 평균 수익률 (분리도가 클수록 신호가 유효)</b><br><br>
                • 매칭 결과가 <b style='color:#EF4444;'>매수 신호</b>였던 시점들의 실제 평균 20일 수익률: <b style='color:#EF4444;'>{bullish_avg:+.2f}%</b> ({int(bullish_mask.sum())}회)<br>
                • 매칭 결과가 <b style='color:#3B82F6;'>매도 신호</b>였던 시점들의 실제 평균 20일 수익률: <b style='color:#3B82F6;'>{bearish_avg:+.2f}%</b> ({int(bearish_mask.sum())}회)
            </div>
            """, unsafe_allow_html=True)

            wf_fig = go.Figure()
            colors = ["#EF4444" if r["hit"] else "#6B7280" for r in wf_results]
            wf_fig.add_trace(go.Scatter(
                x=predicted, y=actual, mode="markers",
                marker=dict(size=10, color=colors),
                text=[r["date"].strftime("%Y-%m-%d") for r in wf_results],
                hovertemplate="%{text}<br>예측: %{x:.2f}%<br>실제: %{y:.2f}%<extra></extra>"
            ))
            wf_fig.add_hline(y=0, line=dict(color="#4B5563", dash="dot"))
            wf_fig.add_vline(x=0, line=dict(color="#4B5563", dash="dot"))
            wf_fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=30, b=10),
                title=dict(text="예측 수익률 vs 실제 수익률 (붉은 점 = 방향 적중)", font=dict(color="#D1D5DB", size=13)),
                xaxis=dict(title="예측 평균 수익률(%)", showgrid=True, gridcolor='#374151', title_font=dict(color="#9CA3AF"), tickfont=dict(color="#9CA3AF")),
                yaxis=dict(title="실제 20일 수익률(%)", showgrid=True, gridcolor='#374151', title_font=dict(color="#9CA3AF"), tickfont=dict(color="#9CA3AF")),
            )
            st.plotly_chart(wf_fig, use_container_width=True)
