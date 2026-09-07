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

# [수정] 국면 개수별 라벨 체계를 분리 정의.
# 지난 롤링vs확장 교차검증에서 3국면 중 가운데(Caution)만 유의성이 불안정했고,
# 극단 두 국면(Safe/Danger)은 항상 견고하게 유의미했다. 그래서 2국면을 기본값으로 승격하고,
# 중간 라벨 없이 "안정/위험"으로 직접 대비되게 재설계한다. 3·4국면도 필요시 선택 가능하도록 유지.
LABEL_SETS = {
    2: [("🟢 안정 국면 (Safe)", "green"),
        ("🔴 위험 국면 (Danger)", "red")],
    3: [("🟢 상승/안정 국면 (Safe)", "green"),
        ("🟡 변동성 확대 (Caution)", "#B8B800"),
        ("🔴 위험 국면 (Danger)", "red")],
    4: [("🟢 상승/안정 국면 (Safe)", "green"),
        ("🟡 변동성 확대 (Caution)", "#B8B800"),
        ("🟠 조정 국면 (Warning)", "orange"),
        ("🔴 공포/폭락 국면 (Danger)", "red")],
}

# --- 사이드바 설정 ---
with st.sidebar:
    st.header("⚙️ 모델 설정")
    # [최적화] 기본값 10->7년: 데이터 길이가 재학습 96회 각각의 학습 데이터 크기에 직결됨
    lookback_years = st.slider("데이터 수집 기간 (년)", min_value=5, max_value=15, value=7)
    # [수정] 기본값 3->2: 교차검증 결과 중간 국면(Caution)이 롤링/확장에 따라 유의성이
    # 왔다갔다했고, 극단 2개(Safe/Danger)만 두 방식 모두에서 견고하게 유의했음.
    n_states = st.slider("국면(Regime) 개수", min_value=2, max_value=4, value=2)
    if n_states == 2:
        st.caption("💡 2국면(안정/위험)을 기본 권장합니다 — 교차검증에서 가장 견고했던 조합입니다.")

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
        # [최적화] 기본값 21->63: 재학습 횟수가 4.5배 이상 줄어듦(약 96회 -> 약 20회대).
        # 국면 판독의 민감도가 떨어지는 대신, 최초 실행 시 CPU 부하가 크게 줄어듦.
        retrain_freq = st.select_slider(
            "재학습 주기 (거래일)", options=[5, 10, 21, 63, 126],
            value=63, help="21=약 1개월, 63=약 1분기, 126=약 반기"
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
    """rank(0=안정 ... n-1=위험) -> (라벨, 색상) 매핑. LABEL_SETS에 없는 n이면 4국면 세트를 재활용."""
    labels = LABEL_SETS.get(n_components, LABEL_SETS[4])
    return {rank: labels[rank] for rank in range(n_components)}


# --- 2-A. 전체기간 일괄학습 (In-sample, 설명/참고용) ---
@st.cache_resource
def fit_hmm_insample(df: pd.DataFrame, n_components: int):
    feature_cols = get_feature_cols(df)
    X_raw = df[feature_cols].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # [최적화] "full"->"diag": in-sample 모드는 1회만 학습하니 원래도 가볍지만,
    # 두 모드의 국면 정의 방식을 일치시켜야 in-sample vs 워크포워드 비교가 의미 있음
    model = GaussianHMM(n_components=n_components, covariance_type="diag",
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
                # [최적화] covariance_type "full"->"diag": 6개 피처 간 완전 공분산 행렬을
                # 96회 재학습마다 반복 역산하는 게 CPU 사용의 가장 큰 비중을 차지했음.
                # diag는 피처 간 상관은 반영 못 하지만 연산량이 훨씬 가벼움.
                # n_iter도 100->50으로 낮춤: tol=1e-3 기준으로는 대부분 그 전에 수렴함.
                candidate = GaussianHMM(n_components=n_components, covariance_type="diag",
                                         n_iter=50, random_state=42, tol=1e-3)
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
            미래5일평균=("Fwd_Ret_5", "mean"),
            미래5일표본수=("Fwd_Ret_5", "count"),
            미래20일평균=("Fwd_Ret_20", "mean"),
            미래20일중앙값=("Fwd_Ret_20", "median"),
            미래20일최솟값=("Fwd_Ret_20", "min"),
            미래20일표본수=("Fwd_Ret_20", "count"),
        )
        summary.index = [state_map[i][0] for i in summary.index]
        st.dataframe(summary.style.format({
            "당일동시성수익률_참고용": "{:.4%}",
            "평균VIX": "{:.2f}",
            "미래5일평균": "{:.4%}",
            "미래20일평균": "{:.4%}",
            "미래20일중앙값": "{:.4%}",
            "미래20일최솟값": "{:.4%}",
        }))
        st.caption(
            "💡 **평균만 보지 마세요**: '미래20일최솟값'이 크게 마이너스인데 '미래20일평균'이 플러스라면, "
            "그 국면은 대체로는 무난하다가 가끔 크게 오르는 게 아니라 '가끔 크게 오르는 소수의 사건이 "
            "평균을 끌어올린' 구조일 수 있습니다. 중앙값이 평균보다 훨씬 낮다면 특히 의심해 보세요."
        )

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
                          .map(_highlight_sig, subset=["p-value(5일)", "p-value(20일)"])
        )
        st.caption(
            "p-value < 0.05(초록 강조)면 그 국면의 미래 수익률 분포가 나머지 국면 전체와 통계적으로 "
            "유의미하게 다르다는 뜻입니다. 표본수가 적은 국면(특히 롤링 윈도우를 짧게 잡거나 국면 개수를 "
            "늘렸을 때)은 검정력이 낮아 유의성이 잘 안 나올 수 있으니 표본수 컬럼을 함께 확인하세요. "
            "또한 국면 개수(n_states)를 조정할 때마다 결과가 크게 흔들린다면, 그 자체가 국면 구분의 "
            "안정성이 낮다는 신호입니다."
        )

        # --- [추가] 독립표본(비중첩) 재검정: 겹치는 20일 윈도우로 인한 표본 부풀림 보정 ---
        st.markdown("##### 📏 독립표본(비중첩) 재검정 — 표본 부풀림 보정")
        st.caption(
            "⚠️ 위 검정은 미래 20일 수익률을 하루씩 밀려가며(겹치게) 계산한 값을 그대로 사용했습니다. "
            "예를 들어 표본 1,000개라도 이웃한 표본끼리 19일치 데이터가 겹치므로, 실제로 독립적인 "
            "정보량은 그보다 훨씬 적고 p-value는 실제보다 낙관적으로(과대) 나올 수 있습니다. "
            "아래는 20거래일 간격으로 서로 안 겹치게 골라낸 표본만으로 같은 검정을 다시 수행한 결과입니다 "
            "— 표본수는 크게 줄지만, 훨씬 현실적인(보수적인) 유의성 추정치입니다."
        )

        nonoverlap_df = analyzed_df.iloc[::20]  # 20거래일 간격 = 서로 겹치지 않는 표본만 추출

        nonoverlap_rows = []
        for r in sorted(nonoverlap_df["Regime"].unique()):
            label = state_map[r][0]
            this_20 = nonoverlap_df.loc[nonoverlap_df["Regime"] == r, "Fwd_Ret_20"].dropna()
            rest_20 = nonoverlap_df.loc[nonoverlap_df["Regime"] != r, "Fwd_Ret_20"].dropna()

            row = {
                "국면": label,
                "독립표본수": len(this_20),
                "평균(독립표본)": this_20.mean() if len(this_20) > 0 else np.nan,
                "중앙값(독립표본)": this_20.median() if len(this_20) > 0 else np.nan,
            }
            if len(this_20) >= 5 and len(rest_20) >= 5:
                try:
                    _, p = mannwhitneyu(this_20, rest_20, alternative="two-sided")
                    row["p-value(독립표본)"] = p
                except Exception:
                    row["p-value(독립표본)"] = np.nan
            else:
                row["p-value(독립표본)"] = np.nan
            nonoverlap_rows.append(row)

        nonoverlap_sig_df = pd.DataFrame(nonoverlap_rows).set_index("국면")
        st.dataframe(
            nonoverlap_sig_df.style.format({
                "평균(독립표본)": "{:.2%}",
                "중앙값(독립표본)": "{:.2%}",
                "p-value(독립표본)": "{:.4f}",
            }, na_rep="—").map(_highlight_sig, subset=["p-value(독립표본)"])
        )
        st.caption(
            "여기서도 p<0.05가 유지된다면, 위 전체표본 검정 결과가 표본 부풀림에 의한 착시가 아니라는 "
            "뜻입니다. 반대로 전체표본에서는 유의했는데 여기서 유의성을 잃는다면, 원래 결과는 겹치는 "
            "윈도우 때문에 과대평가됐을 가능성이 높으니 그 국면 신호는 보수적으로 취급하세요. "
            "(참고: 20일 간격 중 임의의 한 시작점만 뽑은 결과라, 시작점을 바꾸면 표본 구성도 달라집니다 "
            "— 정확한 값이라기보다 대략적인 규모 확인용으로 보십시오. 완전한 확인은 바로 아래 "
            "다중 시작점 검정을 참고하세요.)"
        )

        # --- [추가] 다중 시작점(Phase) 검정: 위 단일 시작점(0번째) 결과가 우연이 아닌지 확인 ---
        st.markdown("##### 🔬 다중 시작점(Phase) 독립표본 재검정")
        st.caption(
            "바로 위 결과는 20일 간격 중 시작점 하나(0번째 거래일)만 골라 계산한 값입니다. "
            "시작점을 0~19번째 거래일로 바꿔가며 20가지 조합 전부에 대해 같은 검정을 반복하면, "
            "특정 시작점 하나의 우연에 결과가 좌우된 건 아닌지 확인할 수 있습니다."
        )

        phase_pvalues = {r: [] for r in sorted(analyzed_df["Regime"].unique())}
        for offset in range(20):
            phase_df = analyzed_df.iloc[offset::20]
            for r in phase_pvalues.keys():
                this_p = phase_df.loc[phase_df["Regime"] == r, "Fwd_Ret_20"].dropna()
                rest_p = phase_df.loc[phase_df["Regime"] != r, "Fwd_Ret_20"].dropna()
                if len(this_p) >= 5 and len(rest_p) >= 5:
                    try:
                        _, p = mannwhitneyu(this_p, rest_p, alternative="two-sided")
                        phase_pvalues[r].append(p)
                    except Exception:
                        pass

        phase_rows = []
        for r, plist in phase_pvalues.items():
            label = state_map[r][0]
            if len(plist) == 0:
                phase_rows.append({
                    "국면": label, "유효 시작점 수": 0, "평균p값": np.nan,
                    "중앙값p값": np.nan, "최댓값p값": np.nan, "p<0.05 비율": np.nan,
                })
                continue
            arr = np.array(plist)
            phase_rows.append({
                "국면": label,
                "유효 시작점 수": len(arr),
                "평균p값": arr.mean(),
                "중앙값p값": np.median(arr),
                "최댓값p값": arr.max(),
                "p<0.05 비율": (arr < 0.05).mean(),
            })

        phase_summary_df = pd.DataFrame(phase_rows).set_index("국면")
        st.dataframe(phase_summary_df.style.format({
            "평균p값": "{:.4f}", "중앙값p값": "{:.4f}", "최댓값p값": "{:.4f}", "p<0.05 비율": "{:.0%}",
        }, na_rep="—"))
        st.caption(
            "'p<0.05 비율'이 100%에 가까울수록 어느 시작점을 고르든 결과가 유의미하다는 뜻이라 "
            "신뢰도가 높습니다. 이 비율이 낮거나 '최댓값p값'이 0.05를 크게 웃돈다면, 특정 시작점에서만 "
            "우연히 유의했던 결과일 수 있으니 그 국면 신호는 보수적으로 취급하세요."
        )

        # --- [추가] 위기 구간별 분해: 전체기간 검정 결과가 특정 사건 하나에 쏠린 게 아닌지 확인 ---
        st.markdown("##### 🗓️ 주요 위기 구간별 국면 분해 (한 사건에 결과가 쏠려 있는지 확인)")
        st.caption(
            "위 전체기간 검정이 유의미해도, 그게 사실은 2020년 코로나 반등 한 번 같은 단일 사건이 "
            "통계를 지배한 결과일 수 있습니다. 아래 세 위기 구간 각각에서 국면별 미래 20일 수익률이 "
            "실제로 비슷한 방향으로 나오는지 개별 확인하세요. 세 구간에서 결과가 들쭉날쭉하다면 "
            "'일반적으로 성립하는 패턴'이 아니라 '특정 사건에 대한 우연한 적합'일 가능성이 있습니다."
        )

        EVENT_WINDOWS = {
            "2018년 4분기 조정": ("2018-10-01", "2018-12-31"),
            "2020년 코로나 폭락": ("2020-02-15", "2020-04-30"),
            "2022년 약세장": ("2022-01-01", "2022-10-31"),
        }

        event_rows = []
        for event_name, (ev_start, ev_end) in EVENT_WINDOWS.items():
            mask = (analyzed_df.index >= ev_start) & (analyzed_df.index <= ev_end)
            ev_df = analyzed_df.loc[mask]
            if ev_df.empty:
                event_rows.append({
                    "사건": event_name, "국면": "(데이터 없음 — 데이터 수집기간/롤링윈도우 범위 밖)",
                    "표본수": 0, "평균": np.nan, "중앙값": np.nan, "최솟값": np.nan,
                })
                continue
            for r in sorted(ev_df["Regime"].unique()):
                vals = ev_df.loc[ev_df["Regime"] == r, "Fwd_Ret_20"].dropna()
                if len(vals) == 0:
                    continue
                event_rows.append({
                    "사건": event_name,
                    "국면": state_map[r][0],
                    "표본수": len(vals),
                    "평균": vals.mean(),
                    "중앙값": vals.median(),
                    "최솟값": vals.min(),
                })

        event_df = pd.DataFrame(event_rows)
        if not event_df.empty:
            st.dataframe(
                event_df.set_index(["사건", "국면"]).style.format(
                    {"평균": "{:.2%}", "중앙값": "{:.2%}", "최솟값": "{:.2%}"}, na_rep="—"
                )
            )

        # --- [추가] 롤링 vs 확장 교차검증: 같은 설정으로 window_mode만 바꿔 재실행하면 자동 비교 ---
        # 주의: window_years/retrain_freq/min_train_years/decode_context는 워크포워드 모드에서만
        # 존재하는 설정이므로, 일괄학습(in-sample) 모드에서는 이 섹션 자체를 건너뛴다.
        if is_walkforward:
            st.markdown("##### 🔁 롤링 vs 확장 윈도우 교차검증")
            if "regime_validation_cache" not in st.session_state:
                st.session_state["regime_validation_cache"] = {}

            current_mode_label = "롤링" if window_years is not None else "확장"
            cfg_sig = (n_states, retrain_freq, min_train_years, decode_context, lookback_years)
            st.session_state["regime_validation_cache"][(current_mode_label, cfg_sig)] = {
                "summary": summary.copy(),
                "sig_df": sig_df.copy(),
            }

            other_mode_label = "확장" if current_mode_label == "롤링" else "롤링"
            other_key = (other_mode_label, cfg_sig)

            if other_key in st.session_state["regime_validation_cache"]:
                prev = st.session_state["regime_validation_cache"][other_key]
                cur_tbl = pd.DataFrame({
                    f"{current_mode_label}_미래20일평균": summary["미래20일평균"],
                    f"{current_mode_label}_p값(20일)": sig_df["p-value(20일)"],
                })
                prev_tbl = pd.DataFrame({
                    f"{other_mode_label}_미래20일평균": prev["summary"]["미래20일평균"],
                    f"{other_mode_label}_p값(20일)": prev["sig_df"]["p-value(20일)"],
                })
                compare_df = cur_tbl.join(prev_tbl, how="outer")
                st.dataframe(compare_df.style.format({
                    f"{current_mode_label}_미래20일평균": "{:.2%}",
                    f"{other_mode_label}_미래20일평균": "{:.2%}",
                    f"{current_mode_label}_p값(20일)": "{:.4f}",
                    f"{other_mode_label}_p값(20일)": "{:.4f}",
                }, na_rep="—"))
                st.caption(
                    "두 윈도우 방식(롤링/확장)에서 국면별 방향(+/-)과 유의성이 같은 패턴으로 나오면, "
                    "결과가 특정 윈도우 설정에 좌우되지 않는 안정적인 신호라는 뜻입니다. 부호가 뒤집히거나 "
                    "한쪽에서만 유의하다면 윈도우 설정에 결과가 민감하다는 신호이니 신뢰도를 낮춰 보십시오."
                )
            else:
                st.info(
                    f"현재는 '{current_mode_label}' 결과만 있습니다. 사이드바에서 '학습 윈도우'를 "
                    f"'{other_mode_label}'(으)로 바꿔 같은 설정으로 한 번 더 실행하면, 이 자리에 자동으로 "
                    f"두 결과가 나란히 비교되어 나타납니다."
                )
        else:
            st.caption(
                "ℹ️ 롤링/확장 윈도우 교차검증은 워크포워드 모드에서만 제공됩니다. "
                "사이드바에서 '판독 모드'를 '워크포워드'로 바꿔 확인하세요."
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
