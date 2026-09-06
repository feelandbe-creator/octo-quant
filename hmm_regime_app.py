import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 페이지 기본 설정
st.set_page_config(page_title="Wall St. HMM Regime Engine", layout="wide")
st.title("🛡️ Wall Street HMM Regime Switching Model (V4 워크포워드)")
st.caption(f"판독 기준 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (KST)")

FEATURE_CANDIDATES = ["Return", "VIX_Level", "TNX_Diff", "Yield_Curve", "Credit_Stress", "Gold_Rel"]
PALETTE = ["green", "#B8B800", "orange", "red"]
LABELS = ["🟢 상승/안정 국면 (Safe)", "🟡 변동성 확대 (Caution)",
          "🟠 조정 국면 (Warning)", "🔴 공포/폭락 국면 (Danger)"]

# --- 사이드바 설정 ---
with st.sidebar:
    st.header("⚙️ 모델 설정")
    lookback_years = st.slider("데이터 수집 기간 (년)", min_value=5, max_value=15, value=10)
    n_states = st.slider("국면(Regime) 개수", min_value=2, max_value=4, value=3)

    st.divider()
    st.header("🧪 학습 방식")
    mode = st.radio(
        "판독 모드 선택",
        ["워크포워드 (실전용, 미래참조 없음)", "전체기간 일괄학습 (설명/참고용, in-sample)"],
        index=0,
    )
    is_walkforward = mode.startswith("워크포워드")

    if is_walkforward:
        st.caption("매 재학습 시점까지의 데이터만 사용하여, 과거 판독 결과에 미래 정보가 섞이지 않습니다.")
        retrain_freq = st.select_slider(
            "재학습 주기 (거래일)", options=[5, 10, 21, 63, 126],
            value=21, help="21=약 1개월, 63=약 1분기, 126=약 반기"
        )
        window_mode = st.radio("학습 윈도우", ["롤링(최근 N년)", "확장(전체 누적)"], index=0)
        window_years = None
        if window_mode.startswith("롤링"):
            window_years = st.slider("롤링 윈도우 (년)", min_value=2, max_value=10, value=5)
        min_train_years = st.slider("최소 워밍업 기간 (년)", min_value=1, max_value=5, value=2)
        decode_context = st.slider("디코딩 컨텍스트 (거래일)", min_value=30, max_value=252, value=120,
                                    help="당일 국면 판정 시 참고하는 직전 데이터 길이. 미래 데이터는 절대 포함되지 않습니다.")
        st.warning("⏱️ 워크포워드 모드는 여러 번 재학습을 반복하므로 첫 로딩에 다소 시간이 걸릴 수 있습니다.")


# --- 1. 데이터 수집 함수 (SPY, VIX, TNX, IRX, 신용스프레드, 달러 통합 수집) ---
@st.cache_data(ttl=3600)
def fetch_macro_data(years: int):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years)

    tickers = {
        "SPY": "SPY",      # S&P500 (위험자산 가격)
        "VIX": "^VIX",     # 변동성/공포지수
        "TNX": "^TNX",     # 미 10년물 국채금리
        "IRX": "^IRX",     # 미 13주(3개월) 국채금리 -> 장단기 금리차(경기침체 신호) 산출용
        "HYG": "HYG",      # 하이일드 회사채 ETF
        "LQD": "LQD",      # 투자등급 회사채 ETF (HYG/LQD 비율 = 신용 스프레드 프록시)
        "GLD": "GLD",      # 금 (안전자산 선호 심리)
        "DXY": "DX-Y.NYB", # 달러 인덱스 (글로벌 유동성/강달러 압력)
    }

    raw = yf.download(
        list(tickers.values()),
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=True,
        group_by="column",
    )

    if raw is None or raw.empty:
        raise ValueError("yfinance에서 데이터를 받아오지 못했습니다. 네트워크 상태 또는 티커명을 확인하세요.")

    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw

    df = pd.DataFrame(index=close.index)
    missing = []
    for name, ticker in tickers.items():
        if ticker in close.columns:
            df[name] = close[ticker]
        else:
            missing.append(ticker)

    if missing:
        st.warning(f"다음 티커 데이터를 가져오지 못해 제외합니다: {', '.join(missing)}")

    for required in ["SPY", "VIX", "TNX"]:
        if required not in df.columns:
            raise ValueError(f"필수 데이터({required})를 가져오지 못했습니다.")

    df = df.ffill().dropna()

    df["Return"] = df["SPY"].pct_change()
    df["VIX_Level"] = df["VIX"]
    df["TNX_Diff"] = df["TNX"].diff()

    if "IRX" in df.columns:
        df["Yield_Curve"] = df["TNX"] - df["IRX"]
    else:
        df["Yield_Curve"] = np.nan

    if "HYG" in df.columns and "LQD" in df.columns:
        credit_ratio = df["HYG"] / df["LQD"]
        df["Credit_Stress"] = credit_ratio.pct_change()
    else:
        df["Credit_Stress"] = np.nan

    if "GLD" in df.columns:
        df["Gold_Rel"] = df["GLD"].pct_change() - df["Return"]
    else:
        df["Gold_Rel"] = np.nan

    # --- [수정] 순환논리 방지용 "미래(전방) 수익률" 컬럼 추가 ---
    # 주의: 이 두 컬럼은 FEATURE_CANDIDATES에 절대 포함되지 않는다(모델 학습에는 전혀 쓰이지 않음).
    # 오직 "국면 판독이 사후적으로 미래 수익률과 관련이 있는지"를 검증하는 용도로만 사용한다.
    # Fwd_Ret_N(t) = (SPY[t+N] / SPY[t]) - 1  ->  N=5,20 거래일
    df["Fwd_Ret_5"] = df["SPY"].shift(-5) / df["SPY"] - 1
    df["Fwd_Ret_20"] = df["SPY"].shift(-20) / df["SPY"] - 1

    # [수정] 기존에는 df.dropna()로 전체 컬럼 기준 결측치를 제거했는데,
    # 이렇게 하면 Fwd_Ret_5/20이 계산 안 되는 최근 5~20거래일(가장 최신 데이터!)이
    # 통째로 잘려나가 "오늘의 국면 판독"이 불가능해진다.
    # 모델 학습/판독에 실제로 쓰이는 FEATURE_CANDIDATES 기준으로만 결측치를 제거하고,
    # Fwd_Ret 컬럼의 결측치(최근 구간)는 그대로 남겨서 요약 통계에서만 자연히 제외되게 한다.
    df = df.dropna(subset=FEATURE_CANDIDATES)
    return df


def get_feature_cols(df: pd.DataFrame):
    return [c for c in FEATURE_CANDIDATES if c in df.columns and df[c].notna().all()]


def build_state_map(n_components: int):
    """rank(0=안정 ... n-1=공포) -> (라벨, 색상) 매핑. 모든 모드가 공유하는 표시 규칙."""
    state_map = {}
    for rank in range(n_components):
        idx = min(rank, len(LABELS) - 1)
        state_map[rank] = (LABELS[idx], PALETTE[idx])
    return state_map


# --- 2-A. 전체기간 일괄학습 (In-sample, 설명/참고용) ---
@st.cache_resource
def fit_hmm_insample(df: pd.DataFrame, n_components: int):
    feature_cols = get_feature_cols(df)
    X_raw = df[feature_cols].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    model = GaussianHMM(n_components=n_components, covariance_type="full",
                         n_iter=1000, random_state=42, tol=1e-4)
    model.fit(X)
    converged = model.monitor_.converged

    raw_states = model.predict(X)
    df = df.copy()

    # raw_state -> rank(VIX 평균 오름차순) 재매핑
    mean_vix = [df.loc[raw_states == i, "VIX_Level"].mean() for i in range(n_components)]
    rank_of_raw = {raw: rank for rank, raw in enumerate(np.argsort(mean_vix))}
    df["Regime"] = [rank_of_raw[s] for s in raw_states]

    return df, feature_cols, converged, 1, 1


# --- 2-B. 워크포워드 (실전용, look-ahead 없음) ---
@st.cache_resource
def fit_hmm_walkforward(df: pd.DataFrame, n_components: int, retrain_freq: int,
                         min_train_days: int, window_days, decode_context: int):
    feature_cols = get_feature_cols(df)
    X_raw = df[feature_cols].values
    T = len(X_raw)

    regimes = np.full(T, -1, dtype=int)
    n_fail = 0
    n_fit = 0

    model = None
    scaler = None
    rank_of_raw = None
    last_retrain = -10**9
    train_start = 0

    progress = st.progress(0, text="워크포워드 재학습 진행 중...")

    for t in range(min_train_days, T):
        need_retrain = (model is None) or (t - last_retrain >= retrain_freq)

        if need_retrain:
            if window_days is not None:
                train_start = max(0, t - window_days)
            else:
                train_start = 0

            train_data = X_raw[train_start:t + 1]  # 오늘까지의 데이터만 사용 (미래 미참조)

            try:
                scaler = StandardScaler().fit(train_data)
                Xs_train = scaler.transform(train_data)
                candidate = GaussianHMM(n_components=n_components, covariance_type="full",
                                         n_iter=100, random_state=42, tol=1e-3)
                candidate.fit(Xs_train)

                # 학습 데이터 내에서의 VIX 평균 기준 rank 매핑 (재학습마다 임의번호를 일관 의미로 재정렬)
                train_raw_states = candidate.predict(Xs_train)
                vix_slice = df["VIX_Level"].values[train_start:t + 1]
                mean_vix = [vix_slice[train_raw_states == i].mean()
                            if np.any(train_raw_states == i) else np.inf
                            for i in range(n_components)]
                rank_of_raw = {raw: rank for rank, raw in enumerate(np.argsort(mean_vix))}

                model = candidate
                last_retrain = t
                n_fit += 1
            except Exception:
                n_fail += 1
                # 재학습 실패 시 직전 모델을 계속 사용 (완전 중단 방지)

            progress.progress(min(1.0, (t - min_train_days + 1) / max(1, T - min_train_days)),
                               text=f"워크포워드 재학습 진행 중... ({t - min_train_days + 1}/{T - min_train_days})")

        if model is None:
            continue

        ctx_start = max(train_start, t - decode_context)
        Xs_ctx = scaler.transform(X_raw[ctx_start:t + 1])
        try:
            decoded = model.predict(Xs_ctx)
            raw_today = decoded[-1]
            regimes[t] = rank_of_raw.get(raw_today, 0)
        except Exception:
            regimes[t] = regimes[t - 1] if t > 0 else 0

    progress.empty()

    df = df.copy()
    df["Regime"] = regimes
    df = df.iloc[min_train_days:]  # 워밍업 구간(모델 없음) 제외
    df = df[df["Regime"] >= 0]

    return df, feature_cols, n_fail, n_fit, min_train_days


# --- 3. 메인 로직 및 화면 출력 ---
try:
    with st.spinner("거시 데이터 수집 중 (가격/VIX/금리/신용/금)..."):
        macro_df = fetch_macro_data(lookback_years)

    if is_walkforward:
        min_train_days = int(min_train_years * 252)
        window_days = int(window_years * 252) if window_years else None

        if len(macro_df) <= min_train_days + retrain_freq:
            st.error("데이터 기간이 워크포워드 워밍업 기간보다 짧습니다. 데이터 수집 기간을 늘리거나 워밍업 기간을 줄여주세요.")
            st.stop()

        analyzed_df, used_features, n_fail, n_fit, warmup = fit_hmm_walkforward(
            macro_df, n_states, retrain_freq, min_train_days, window_days, decode_context
        )
        state_map = build_state_map(n_states)

        st.caption(f"워크포워드: 총 {n_fit}회 재학습 수행 (워밍업 {warmup}거래일 제외 후 시작)"
                   + (f", 실패 {n_fail}건은 직전 모델로 대체" if n_fail else ""))
    else:
        analyzed_df, used_features, converged, n_fit, warmup = fit_hmm_insample(macro_df, n_states)
        state_map = build_state_map(n_states)
        if not converged:
            st.warning("⚠️ HMM 모델이 완전히 수렴하지 않았습니다. 결과 해석에 참고하세요.")
        st.caption("⚠️ 이 모드는 전체 기간을 한 번에 학습한 in-sample 결과이며, 과거 국면 판독에 미래 데이터의 영향이 섞여 있을 수 있습니다.")

    current_state_idx = analyzed_df["Regime"].iloc[-1]
    current_state_info = state_map[current_state_idx]

    last_spy = analyzed_df["SPY"].iloc[-1]
    last_vix = analyzed_df["VIX"].iloc[-1]
    last_tnx = analyzed_df["TNX"].iloc[-1]

    st.subheader("📊 현재 거시 경제 (Macro) 투심 판독 결과")

    cols = st.columns(5)
    cols[0].metric("S&P 500 (SPY)", f"${last_spy:.2f}", f"{analyzed_df['Return'].iloc[-1]*100:.2f}%")
    cols[1].metric("VIX (공포지수)", f"{last_vix:.2f}", "30 이상 = 극도의 공포", delta_color="inverse")
    cols[2].metric("미 10년물 금리", f"{last_tnx:.2f}%", f"{analyzed_df['TNX_Diff'].iloc[-1]:.3f}%p", delta_color="inverse")
    if "Yield_Curve" in analyzed_df.columns:
        cols[3].metric("장단기 금리차 (10Y-3M)", f"{analyzed_df['Yield_Curve'].iloc[-1]:.2f}%p",
                        "음수=침체 경고", delta_color="off")
    if "Credit_Stress" in analyzed_df.columns:
        cols[4].metric("신용 스프레드 변화 (HYG/LQD)", f"{analyzed_df['Credit_Stress'].iloc[-1]*100:.2f}%",
                        "하락=신용경색", delta_color="off")

    st.markdown(f"""
    <div style="padding: 20px; border-radius: 10px; background-color: {current_state_info[1]}; color: white; text-align: center;">
        <h2 style="margin: 0;">현재 AI 판독 국면: {current_state_info[0]}</h2>
        <p style="margin-top: 10px; font-size: 16px;">모드: {mode} · 사용된 피처: {', '.join(used_features)}</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # --- 4. 시각화 (배경 음영으로 국면 구간 표시) ---
    st.subheader("📈 주가 흐름 및 AI 국면 감지 히스토리")

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(analyzed_df.index, analyzed_df["SPY"], color="black", label="SPY Price", linewidth=1)

    regimes = analyzed_df["Regime"].values
    dates = analyzed_df.index
    seen_labels = set()
    start_idx = 0
    for i in range(1, len(regimes) + 1):
        if i == len(regimes) or regimes[i] != regimes[start_idx]:
            r = regimes[start_idx]
            label, color = state_map[r]
            ax.axvspan(dates[start_idx], dates[i - 1], color=color, alpha=0.25,
                       label=label if label not in seen_labels else None)
            seen_labels.add(label)
            start_idx = i

    ax.set_title("S&P 500 Price & AI Detected Regimes", fontsize=16)
    ax.set_xlabel("Date")
    ax.set_ylabel("SPY Price (USD)")
    handles, labels_ = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper left")
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)

    # --- [수정] 국면별 통계 요약: 순환논리(당일 Return) 대신 미래(D+5/D+20) 수익률 기반 검증 ---
    with st.expander("🔎 국면별 통계 요약 보기 (사후 예측력 검증)"):
        st.caption(
            "⚠️ **읽는 법**: '당일 동시성 수익률'은 HMM이 국면을 나눌 때 실제로 사용한 값(Return) "
            "그 자체이기 때문에, 국면별로 다르게 나오는 게 당연합니다(순환논리 — 예측력의 증거가 "
            "될 수 없음). 실제 예측력 판단은 아래 '미래 5일/20일 수익률'을 보세요 — 이 값은 국면 "
            "판독 시점 **이후**의, 모델이 학습 때 전혀 보지 못한 미래 데이터입니다."
        )

        summary = analyzed_df.groupby("Regime").agg(
            일수=("SPY", "count"),
            당일동시성수익률_참고용=("Return", "mean"),
            평균VIX=("VIX_Level", "mean"),
            미래5일수익률=("Fwd_Ret_5", "mean"),
            미래5일표본수=("Fwd_Ret_5", "count"),
            미래20일수익률=("Fwd_Ret_20", "mean"),
            미래20일표본수=("Fwd_Ret_20", "count"),
        )
        summary.index = [state_map[i][0] for i in summary.index]
        st.dataframe(summary.style.format({
            "당일동시성수익률_참고용": "{:.4%}",
            "평균VIX": "{:.2f}",
            "미래5일수익률": "{:.4%}",
            "미래20일수익률": "{:.4%}",
        }))

        st.markdown("##### 📐 국면별 예측력 유의성 검정 (Mann-Whitney U: 이 국면 vs 나머지 전체)")

        sig_rows = []
        for r in sorted(analyzed_df["Regime"].unique()):
            label = state_map[r][0]
            this_5 = analyzed_df.loc[analyzed_df["Regime"] == r, "Fwd_Ret_5"].dropna()
            rest_5 = analyzed_df.loc[analyzed_df["Regime"] != r, "Fwd_Ret_5"].dropna()
            this_20 = analyzed_df.loc[analyzed_df["Regime"] == r, "Fwd_Ret_20"].dropna()
            rest_20 = analyzed_df.loc[analyzed_df["Regime"] != r, "Fwd_Ret_20"].dropna()

            row = {"국면": label, "표본수(5일)": len(this_5), "표본수(20일)": len(this_20)}

            if len(this_5) >= 5 and len(rest_5) >= 5:
                try:
                    _, p5 = mannwhitneyu(this_5, rest_5, alternative="two-sided")
                    row["p-value(5일)"] = p5
                except Exception:
                    row["p-value(5일)"] = np.nan
            else:
                row["p-value(5일)"] = np.nan

            if len(this_20) >= 5 and len(rest_20) >= 5:
                try:
                    _, p20 = mannwhitneyu(this_20, rest_20, alternative="two-sided")
                    row["p-value(20일)"] = p20
                except Exception:
                    row["p-value(20일)"] = np.nan
            else:
                row["p-value(20일)"] = np.nan

            sig_rows.append(row)

        sig_df = pd.DataFrame(sig_rows).set_index("국면")

        def _highlight_sig(val):
            if pd.isna(val):
                return ""
            return "color: #16a34a; font-weight: 700;" if val < 0.05 else ""

        st.dataframe(
            sig_df.style.format({"p-value(5일)": "{:.4f}", "p-value(20일)": "{:.4f}"})
                          .applymap(_highlight_sig, subset=["p-value(5일)", "p-value(20일)"])
        )
        st.caption(
            "p-value < 0.05(초록 강조)면 그 국면의 미래 수익률 분포가 나머지 국면 전체와 통계적으로 "
            "유의미하게 다르다는 뜻입니다. 표본수가 적은 국면(특히 롤링 윈도우를 짧게 잡거나 국면 개수를 "
            "늘렸을 때)은 검정력이 낮아 유의성이 잘 안 나올 수 있으니 표본수 컬럼을 함께 확인하세요. "
            "또한 국면 개수(n_states)를 조정할 때마다 결과가 크게 흔들린다면, 그 자체가 국면 구분의 "
            "안정성이 낮다는 신호입니다."
        )

    if is_walkforward:
        st.info("""
        **💡 워크포워드 모드:** 매 재학습 시점까지의 데이터로만 모델을 학습하고, 당일 국면 판정 시에도
        직전 컨텍스트 구간까지의 정보만 사용합니다. 즉 차트상 과거 어떤 날짜의 색상도 그 날짜 이후의 데이터로부터
        영향을 받지 않습니다 (look-ahead bias 제거). 실전 매매 판단에 참고하기에 적합한 모드입니다.
        단, 재학습 주기·윈도우 길이·컨텍스트 길이 설정에 따라 결과가 달라질 수 있으니 여러 설정으로 민감도를 확인해보세요.
        국면 판독 자체의 신뢰도는 위 '국면별 통계 요약'의 미래(D+5/D+20) 수익률 유의성 검정으로 확인하십시오.
        """)
    else:
        st.info("""
        **💡 일괄학습 모드:** 전체 기간을 한 번에 학습하므로 국면 경계가 매끄럽고 설명하기 좋지만,
        과거 시점의 국면 판정에 미래 데이터의 영향이 섞여 있어 실전 신호로 쓰기에는 낙관적으로 보일 수 있습니다.
        실전 검증은 '워크포워드' 모드로 확인하십시오.
        """)

except Exception as e:
    st.error(f"데이터 처리 중 에러가 발생했습니다: {str(e)}")
    st.write("yfinance 데이터 로드 지연이거나 일부 티커(DX-Y.NYB 등)가 일시적으로 응답하지 않을 수 있습니다. "
              "우측 상단 점 3개 → 'Clear cache' 후 재시도해 주세요.")
