import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

try:
    import statsmodels.api as sm
    _STATSMODELS_AVAILABLE = True
except ImportError:
    _STATSMODELS_AVAILABLE = False

# 페이지 기본 설정
st.set_page_config(page_title="Wall St. HMM Regime Engine", layout="wide")

# --- [추가] 상단 타이틀 + 우측 매뉴얼 버튼 ---
title_col, manual_col = st.columns([6, 1])
with title_col:
    st.title("🛡️ Wall Street HMM Regime Switching Model (V4 워크포워드)")
    st.caption(f"판독 기준 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (KST)")
with manual_col:
    st.write("")  # 타이틀과 수직 정렬 맞추기용 여백
    with st.popover("📖 사용 설명서", use_container_width=True):
        st.markdown("""
### 이 앱의 목적
미국 증시(S&P500) 관련 6개 거시지표(수익률·VIX·금리차분·장단기금리차·신용스프레드·금 상대수익률)를
HMM(은닉마르코프모델)에 학습시켜, 현재 시장이 "안정" 국면인지 "위험" 국면인지 판독합니다.

**⚠️ 이 앱은 자동 매매 신호가 아닙니다.** 검증 결과, 국면 판정이 미래 수익률과 통계적으로
약~중간 강도로 연관되어 있다는 근거는 있지만, 단독으로 매수·매도를 결정할 만큼 강력하지
않습니다. 다른 매매 판단(예: 리스크 통제 센터, 손절/익절 로직)을 내리기 **전에 한 번 더
참고하는 방어적 지표(범퍼)**로 활용하는 것을 권장합니다.

### 사이드바 사용법
- **데이터 수집 기간**: 학습에 사용할 과거 데이터 길이(년). 길수록 2008년 금융위기 같은
  장기 약세장까지 포함되어 검증이 더 엄밀해지지만, 계산 시간이 늘어납니다.
- **국면(Regime) 개수**: 기본값 2(안정/위험)를 권장합니다. 교차검증 결과 중간 국면은
  판정이 불안정했습니다.
- **판독 모드**: '워크포워드'가 실전용입니다(미래 데이터를 절대 참조하지 않음). '일괄학습'은
  설명/비교 참고용이며 낙관적으로 보일 수 있습니다.
- **재학습 주기 / 학습 윈도우 / 워밍업 기간 / 디코딩 컨텍스트**: 워크포워드 모드의 세부
  설정입니다. 값을 바꾼 뒤에는 반드시 **"🚀 설정 적용 및 실행"** 버튼을 눌러야 반영됩니다.

### 결과 화면 읽는 법
- **현재 AI 판독 국면 박스**: 오늘의 국면과, 그 판정에 대한 모델의 확신도(%)에 따라
  달라지는 참고용 대응 문구가 함께 표시됩니다. 문구 아래 6칸 표에서 지금 어느 칸에
  해당하는지 색으로 강조됩니다.
- **국면별 통계 요약 (사후 예측력 검증)**: 이 국면 판정이 실제로 미래 수익률과 유의미하게
  연관되는지를 여러 통계적 방법(Mann-Whitney U 검정, 독립표본 재검정, 다중 시작점 검정,
  Newey-West HAC 보정 검정)으로 교차 확인한 결과입니다. 방법마다 결론이 다를 수 있으니
  여러 검정을 함께 보고 신중하게 판단하세요.

### 알려진 한계
- HMM은 후행지표라, 위기 발생 직전까지도 "안정"으로 판정하는 경우가 있었습니다.
- "위험 국면이 안정 국면보다 미래수익률이 낫다"는 패턴이 검증됐지만, 이는 변동성
  평균회귀라는 잘 알려진 현상에 가깝고 강력한 알파는 아닙니다.
- 재학습 주기를 길게 잡을수록(연산 부담은 줄지만) 국면 전환 반응이 최대 그 기간만큼
  늦어질 수 있습니다.
""")

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
# [수정] st.form으로 감싸서, 슬라이더를 여러 개 바꿔도 즉시 재실행되지 않고
# "적용 및 실행" 버튼을 눌러야 한 번에 반영되도록 변경.
with st.sidebar:
    st.header("⚙️ 모델 설정")

    with st.form("settings_form"):
        # [최적화] 기본값 10->7년: 데이터 길이가 재학습 96회 각각의 학습 데이터 크기에 직결됨
        lookback_years_input = st.slider(
            "데이터 수집 기간 (년)", min_value=5, max_value=15, value=7,
            help="""
**무엇을 조절하나요?**
HMM이 학습에 쓸 과거 데이터의 기간(년)입니다. 오늘부터 거슬러 올라가 이만큼의 일별 시세를 수집합니다.

**숫자별 의미**
- 5~7년: 최근 시장 위주. 계산이 빠르지만 2008년 금융위기 같은 극단적 약세장은 빠집니다.
- 10~15년: 2018년 조정·2020년 코로나 폭락·2022년 약세장까지 포함되어 검증이 더 엄밀해집니다.

**늘리면?** 통계 검증의 표본이 많아져 신뢰도는 오르지만, 다운로드·재학습 시간이 늘어 Streamlit 무료 호스팅의 속도 제한(스로틀)에 걸릴 위험이 커집니다.
**줄이면?** 빠르지만, 최근 몇 년의 특수한 상황(예: 저금리 초강세장)에 결과가 치우칠 수 있습니다.

**실전 팁**: 평소엔 7년으로 빠르게 확인하고, 결과를 실제로 신뢰하기 전엔 15년으로 다시 돌려 장기 약세장에서도 같은 패턴이 유지되는지 확인하세요.
"""
        )
        # [수정] 기본값 3->2: 교차검증 결과 중간 국면(Caution)이 롤링/확장에 따라 유의성이
        # 왔다갔다했고, 극단 2개(Safe/Danger)만 두 방식 모두에서 견고하게 유의했음.
        n_states_input = st.slider(
            "국면(Regime) 개수", min_value=2, max_value=4, value=2,
            help="""
**무엇을 조절하나요?**
시장을 몇 개의 국면(regime)으로 나눌지입니다. HMM이 이 개수만큼 서로 다른 '상태'를 자동으로 찾아냅니다.

**숫자별 의미**
- 2: 안정(Safe) / 위험(Danger) 두 가지만 구분. 가장 단순하고 해석이 명확합니다.
- 3: 안정 / 변동성 확대(Caution) / 위험.
- 4: 안정 / Caution / Warning / Danger. 가장 세분화.

**늘리면?** 구분이 세밀해 보이지만, 지난 검증에서 중간 국면(Caution/Warning)은 롤링·확장 등 설정을 바꾸면 유의성이 왔다갔다 하는 불안정한 결과가 나왔습니다.
**줄이면(2)?** 극단 두 상태만 남아, 실제 검증에서 가장 일관되고 견고한 결과를 보였습니다.

**실전 팁**: 특별한 이유가 없다면 2를 유지하세요. 세분화하고 싶다면 반드시 '롤링 vs 확장 교차검증' 결과로 그 중간 국면이 실제로 안정적인지 직접 확인한 뒤 쓰세요.
"""
        )
        if n_states_input == 2:
            st.caption("💡 2국면(안정/위험)을 기본 권장합니다 — 교차검증에서 가장 견고했던 조합입니다.")

        st.divider()
        st.subheader("🧪 학습 방식")
        mode_input = st.radio(
            "판독 모드 선택",
            ["워크포워드 (실전용, 미래참조 없음)", "전체기간 일괄학습 (설명/참고용, in-sample)"],
            index=0,
            help="""
**무엇을 조절하나요?**
국면 판독을 어떤 방식으로 계산할지입니다.

**옵션별 의미**
- 워크포워드(실전용): 각 시점까지의 데이터만으로 그 시점을 학습해, 과거 판독 결과에 미래 정보가 절대 섞이지 않습니다. 그 날짜에 이 정보만 있었다면 실제로 어떤 판정을 냈을지를 재현합니다.
- 일괄학습(in-sample): 전체 기간을 한 번에 학습합니다. 훨씬 빠르지만, 과거 어떤 날짜를 판정할 때도 사실상 미래를 이미 알고 있는 상태라 실제보다 정확해 보이는 착시가 생깁니다.

**실전 팁**: 실제 판단에는 반드시 '워크포워드'를 쓰세요. '일괄학습'은 두 모드의 차이를 비교하거나, 계산이 빨라 국면 개수 등을 빠르게 실험해볼 때만 참고하십시오.
"""
        )
        is_walkforward_input = mode_input.startswith("워크포워드")

        # 워크포워드 전용 세부설정 - 미리 None으로 초기화(일괄학습 모드에서는 미사용)
        retrain_freq_input = None
        window_mode_input = None
        window_years_input = None
        min_train_years_input = None
        decode_context_input = None

        if is_walkforward_input:
            st.caption("매 재학습 시점까지의 데이터만 사용하여, 과거 판독 결과에 미래 정보가 섞이지 않습니다.")
            # [최적화] 기본값 21->63: 재학습 횟수가 4.5배 이상 줄어듦(약 96회 -> 약 20회대).
            retrain_freq_input = st.select_slider(
                "재학습 주기 (거래일)", options=[5, 10, 21, 63, 126],
                value=63,
                help="""
**무엇을 조절하나요?**
워크포워드 모드에서 모델을 얼마나 자주 처음부터 다시 학습시킬지(거래일 기준)입니다.

**숫자별 의미**
- 5~10일: 거의 매주 재학습. 국면 전환에 가장 빠르게 반응하지만 재학습 횟수가 수백 번으로 늘어 계산이 매우 무거워집니다.
- 21일(약 1개월): 반응 속도와 계산 부담의 중간.
- 63일(약 1분기): 계산은 가볍지만, 실제 국면이 바뀐 뒤에도 최대 3개월까지 옛 판정을 유지할 수 있습니다.
- 126일(약 반기): 가장 가볍지만 반응이 가장 느립니다.

**늘리면?** 계산이 빨라져 Streamlit 무료 호스팅의 속도 제한을 피하기 쉬워지지만, 국면 전환을 늦게 알아차립니다.
**줄이면?** 국면 변화에 더 민감하지만, 재학습 횟수가 늘어 속도 제한 위험이 커집니다.

**실전 팁**: 평소 모니터링엔 63일로 가볍게 쓰고, 시장이 급변할 때(위기 뉴스 등)는 21일로 낮춰 더 민감한 판독을 한 번 확인해보세요.
"""
            )
            window_mode_input = st.radio(
                "학습 윈도우", ["롤링(최근 N년)", "확장(전체 누적)"], index=0,
                help="""
**무엇을 조절하나요?**
재학습할 때 얼마나 긴 과거 데이터를 학습에 사용할지의 방식입니다.

**옵션별 의미**
- 롤링(최근 N년): 항상 최근 N년치만 사용합니다. 오래된 과거는 잊혀지고 최근 시장 성격에 더 민감하게 반응합니다.
- 확장(전체 누적): 데이터 수집 시작일부터 오늘까지 전부 누적해서 씁니다. 시간이 지날수록 학습 데이터가 계속 늘어나, 오래된 사건의 영향력이 서서히 옅어지지만 완전히 사라지진 않습니다.

**실전 팁**: 두 방식을 각각 돌려서 결론이 일치하는지 보는 게 가장 안전합니다. 이 앱의 '롤링 vs 확장 교차검증' 표가 그 비교를 자동으로 해줍니다.
"""
            )
            if window_mode_input.startswith("롤링"):
                window_years_input = st.slider(
                    "롤링 윈도우 (년)", min_value=2, max_value=10, value=5,
                    help="""
**무엇을 조절하나요?**
'롤링' 방식일 때, 매번 재학습에 사용할 최근 데이터의 길이(년)입니다.

**숫자별 의미**
- 2~3년: 아주 최근 시장 성격에만 민감. 국면 전환을 빠르게 잡아내지만, 표본이 적어 통계적으로 불안정할 수 있습니다.
- 5년: 균형점 — 최근 흐름을 반영하면서도 충분한 표본을 확보합니다.
- 8~10년: 더 안정적으로 학습되지만, '롤링'의 장점(최신 시장 성격 반영)이 옅어져 사실상 확장 모드와 비슷해집니다.

**실전 팁**: 5년을 기본으로 쓰고, 결과가 지나치게 자주 바뀌어 신뢰하기 어렵다면 7~8년으로 늘려 안정성을 높이세요.
"""
                )
            min_train_years_input = st.slider(
                "최소 워밍업 기간 (년)", min_value=1, max_value=5, value=2,
                help="""
**무엇을 조절하나요?**
모델이 첫 판정을 내리기 전 최소한 이만큼의 데이터를 미리 쌓아두고 시작합니다. 이 기간 동안은 국면 판정 자체가 나오지 않고 차트에서도 제외됩니다.

**숫자별 의미**
- 1년: 빨리 판정을 시작하지만, 학습량이 부족한 채 이른 시점부터 판정을 내려 초반 결과의 신뢰도가 낮을 수 있습니다.
- 2년(기본값): 대체로 안정적인 최소 학습량입니다.
- 3~5년: 더 안정적으로 학습한 후 시작하지만, 그만큼 앱이 실제로 보여주는 판독 기간(차트 구간)이 짧아집니다.

**실전 팁**: '데이터 수집 기간'을 7년으로 짧게 잡았다면 워밍업도 1~2년으로 낮춰야 판독 구간이 충분히 남습니다. 15년으로 길게 잡았다면 워밍업을 2~3년으로 여유 있게 잡아도 됩니다.
"""
            )
            decode_context_input = st.slider(
                "디코딩 컨텍스트 (거래일)", min_value=30, max_value=252, value=120,
                help="""
**무엇을 조절하나요?**
오늘의 국면을 판정할 때, 학습된 모델이 '직전 며칠간의 데이터'를 함께 참고해서 디코딩(국면 추정)할지입니다. 미래 데이터는 절대 포함되지 않습니다 — 항상 과거~오늘까지만입니다.

**숫자별 의미**
- 30~60일: 아주 최근의 단기 흐름만으로 판정. 반응은 빠르지만 하루이틀의 노이즈에도 국면이 쉽게 바뀔 수 있습니다.
- 120일(기본값, 약 반년): 단기 노이즈와 반응 속도의 균형점입니다.
- 200~252일(약 1년): 판정이 더 안정적이고 부드럽게 바뀌지만, 실제 전환 시점보다 판정이 늦게 따라올 수 있습니다.

**실전 팁**: 국면이 하루이틀 사이에도 오락가락한다면 이 값을 늘려 안정성을 높이고, 반대로 반응이 너무 느리다고 느껴지면 줄여서 민감도를 높이세요.
"""
            )
            st.warning("⏱️ 워크포워드 모드는 여러 번 재학습을 반복하므로 첫 로딩에 다소 시간이 걸릴 수 있습니다.")

        st.divider()
        submitted = st.form_submit_button(
            "🚀 설정 적용 및 실행", use_container_width=True, type="primary"
        )
        st.caption("여러 슬라이더를 한꺼번에 바꾼 뒤 이 버튼을 눌러야 반영됩니다.")

    # [추가] 제출된 설정을 session_state에 저장 - 최초 로드 시에는 기본값으로 1회 자동 실행
    if submitted or "applied_settings" not in st.session_state:
        st.session_state["applied_settings"] = {
            "lookback_years": lookback_years_input,
            "n_states": n_states_input,
            "mode": mode_input,
            "is_walkforward": is_walkforward_input,
            "retrain_freq": retrain_freq_input,
            "window_years": window_years_input,
            "min_train_years": min_train_years_input,
            "decode_context": decode_context_input,
        }

# 이후 코드는 기존과 동일한 변수명을 그대로 사용 - "적용된" 설정값에서 꺼내온다
_settings = st.session_state["applied_settings"]
lookback_years = _settings["lookback_years"]
n_states = _settings["n_states"]
mode = _settings["mode"]
is_walkforward = _settings["is_walkforward"]
retrain_freq = _settings["retrain_freq"]
window_years = _settings["window_years"]
min_train_years = _settings["min_train_years"]
decode_context = _settings["decode_context"]


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

    # [추가] 마지막 날(오늘)의 국면 판정 확신도(posterior probability).
    # 참고용 조언 문구를 국면 라벨뿐 아니라 "얼마나 확신하는 판정인지"로도 나누기 위함.
    proba = model.predict_proba(X)
    last_confidence = float(proba[-1, raw_states[-1]])

    return df, feature_cols, converged, 1, 1, last_confidence


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
    last_confidence = None

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
            # [추가] 마지막 거래일(오늘)에 한해서만 확신도 계산 - 매일 계산하면 불필요한 연산.
            if t == T - 1:
                proba_ctx = model.predict_proba(Xs_ctx)
                last_confidence = float(proba_ctx[-1, raw_today])
        except Exception:
            regimes[t] = regimes[t - 1] if t > 0 else 0

    progress.empty()

    df = df.copy()
    df["Regime"] = regimes
    df = df.iloc[min_train_days:]  # 워밍업 구간(모델 없음) 제외
    df = df[df["Regime"] >= 0]

    return df, feature_cols, n_fail, n_fit, min_train_days, last_confidence


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

        analyzed_df, used_features, n_fail, n_fit, warmup, current_confidence = fit_hmm_walkforward(
            macro_df, n_states, retrain_freq, min_train_days, window_days, decode_context
        )
        state_map = build_state_map(n_states)

        st.caption(f"워크포워드: 총 {n_fit}회 재학습 수행 (워밍업 {warmup}거래일 제외 후 시작)"
                   + (f", 실패 {n_fail}건은 직전 모델로 대체" if n_fail else ""))
    else:
        analyzed_df, used_features, converged, n_fit, warmup, current_confidence = fit_hmm_insample(macro_df, n_states)
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

    # --- [추가] 확신도(posterior probability) 기반 참고용 대응 문구 ---
    # 주의: 이건 매매를 자동 실행하는 신호가 아니라, 사람이 판단 전에 한 번 더 확인하라는
    # 참고용 조언입니다. "Danger=매수/매도"식 자동 실행 로직으로 연결하지 마세요.
    is_safest = (current_state_idx == 0)
    is_most_dangerous = (current_state_idx == n_states - 1)
    conf_pct = current_confidence * 100 if current_confidence is not None else None

    if conf_pct is None:
        advisory_title = "ℹ️ 확신도 정보 없음"
        advisory_body = "이번 실행에서는 확신도를 계산하지 못했습니다. 국면 라벨만 참고하세요."
        advisory_color = "#6B7280"
    elif is_safest:
        if conf_pct >= 80:
            advisory_title = f"🟢 안정 국면 · 확신도 {conf_pct:.0f}% (높음)"
            advisory_body = (
                "다만 이 모델은 후행지표입니다 — 실제 위기 발생 직전까지도 '안정'으로 판정된 사례가 "
                "과거 검증에서 여러 번 확인됐습니다(2020년 코로나 직전 등). 이 라벨을 근거로 포지션을 "
                "새로 확대하거나 손절선을 완화하지 말고, 기존 리스크 관리 기준을 그대로 유지하세요."
            )
        elif conf_pct >= 60:
            advisory_title = f"🟢 안정 국면 · 확신도 {conf_pct:.0f}% (중간)"
            advisory_body = (
                "국면 전환 초입 단계일 가능성이 있습니다. VIX·신용스프레드 등 원지표를 평소보다 한 번 "
                "더 직접 확인하고, 이 라벨만으로 안심하는 판단은 피하세요."
            )
        else:
            advisory_title = f"🟢 안정 국면 · 확신도 {conf_pct:.0f}% (낮음 — 사실상 경계선)"
            advisory_body = (
                "국면 판정 자체가 애매한 경계 구간입니다. 이 라벨을 어떤 결정의 근거로도 쓰지 말고, "
                "다른 지표(가격·거래량 추세 등)를 우선하세요."
            )
    elif is_most_dangerous:
        if conf_pct >= 80:
            advisory_title = f"🔴 위험 국면 · 확신도 {conf_pct:.0f}% (높음)"
            advisory_body = (
                "과거 검증에서는 이 국면 진입 후 20일 수익률이 평균적으로 나쁘지 않았습니다(변동성 "
                "평균회귀 경향, 통계적 근거는 중간~중상 수준). 그렇다고 매수 신호로 해석하진 마시고, "
                "다만 이 라벨만 보고 기계적으로 전량 매도하는 성급한 반응은 한 번 더 재검토해보세요."
            )
        elif conf_pct >= 60:
            advisory_title = f"🔴 위험 국면 · 확신도 {conf_pct:.0f}% (중간)"
            advisory_body = (
                "위험 신호가 나왔지만 확신도가 완전하지 않습니다. 리스크 관리 강화는 유효하나, "
                "과잉 대응(급격한 비중 축소 등)은 자제하세요."
            )
        else:
            advisory_title = f"🔴 위험 국면 · 확신도 {conf_pct:.0f}% (낮음 — 사실상 경계선)"
            advisory_body = (
                "국면 판정 자체가 애매한 경계 구간입니다. 이 라벨 하나로 성급한 매도 판단을 내리지 "
                "말고, 다른 지표를 함께 확인하세요."
            )
    else:
        advisory_title = f"🟡 중간 국면 · 확신도 {conf_pct:.0f}%"
        advisory_body = (
            "과거 교차검증에서 중간 국면(Caution/Warning)은 롤링·확장 설정에 따라 유의성이 흔들리는 "
            "구간으로 확인됐습니다. 참고 정도로만 보고, 이 라벨에 큰 의미를 두지 마세요."
        )

    st.markdown(f"""
    <div style="padding: 15px 20px; border-radius: 10px; background-color: rgba(255,255,255,0.04);
                border-left: 4px solid {advisory_color if conf_pct is None else current_state_info[1]};
                margin-top: 10px;">
        <p style="margin: 0; font-weight: 700; font-size: 15px;">{advisory_title}</p>
        <p style="margin-top: 8px; font-size: 14px; color: #9CA3AF; line-height: 1.6;">{advisory_body}</p>
        <p style="margin-top: 8px; font-size: 12px; color: #6B7280;">
            ※ 이 문구는 참고용 조언이며 자동 매매 신호가 아닙니다. 최종 판단은 직접 하세요.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- [추가] 국면 x 확신도 대응표(한눈에 보기) - 현재 해당 칸을 색으로 강조 ---
    _tiers = [("높음 (≥80%)", "high"), ("중간 (60~80%)", "medium"), ("낮음 (<60%)", "low")]

    if conf_pct is None:
        _current_tier = None
    elif conf_pct >= 80:
        _current_tier = "high"
    elif conf_pct >= 60:
        _current_tier = "medium"
    else:
        _current_tier = "low"

    _short_texts = {
        0: {  # 안정(가장 낮은 랭크)
            "high": "방심 금지 · 리스크 유지",
            "medium": "전환초입 가능성 · 재확인",
            "low": "경계선 · 판단 보류",
        },
        n_states - 1: {  # 위험(가장 높은 랭크)
            "high": "반등 잦음(통계) · 매도재검토",
            "medium": "관리강화 · 과잉대응 자제",
            "low": "경계선 · 성급판단 금지",
        },
    }

    def _cell_html(regime_rank, tier_key):
        text = _short_texts.get(regime_rank, {}).get(tier_key, "불안정 구간 · 참고용")
        is_current = (regime_rank == current_state_idx) and (tier_key == _current_tier)
        bg = state_map[regime_rank][1] if is_current else "rgba(255,255,255,0.03)"
        opacity = "1" if is_current else "0.55"
        border = f"2px solid {state_map[regime_rank][1]}" if is_current else "1px solid #374151"
        return (
            f'<td style="padding:10px; border:{border}; background-color:{bg}; opacity:{opacity}; '
            f'font-size:12.5px; text-align:center; color:white;">{text}</td>'
        )

    header_cells = "".join(f'<th style="padding:8px; font-size:12.5px; color:#9CA3AF;">{label}</th>' for label, _ in _tiers)
    body_rows = ""
    for rank in range(n_states):
        row_label = state_map[rank][0]
        row_cells = "".join(_cell_html(rank, tier_key) for _, tier_key in _tiers)
        body_rows += f'<tr><td style="padding:8px; font-size:12.5px; color:#D1D5DB; white-space:nowrap;">{row_label}</td>{row_cells}</tr>'

    st.markdown(f"""
    <div style="margin-top: 14px;">
        <p style="font-size: 13px; color: #9CA3AF; margin-bottom: 6px;">📋 국면 × 확신도 대응표 (현재 판정은 진한 테두리로 표시)</p>
        <table style="width:100%; border-collapse: collapse;">
            <tr><th></th>{header_cells}</tr>
            {body_rows}
        </table>
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

        # --- [추가] Newey-West(HAC) 자기상관 보정 검정: 표본을 버리지 않는 대안 ---
        st.markdown("##### 🧮 Newey-West(HAC) 자기상관 보정 검정 — 표본을 버리지 않는 대안")
        st.caption(
            "위 독립표본 재검정은 표본의 약 95%를 버려서 자기상관을 없앴습니다. Newey-West(HAC, "
            "Heteroskedasticity and Autocorrelation Consistent) 표준오차는 전체 표본을 그대로 쓰면서, "
            "겹치는 윈도우 때문에 생기는 자기상관을 표준오차 계산 단계에서 직접 보정합니다. "
            "국면 더미변수로 회귀분석(y=미래수익률, x=국면 더미)을 돌리고, 그 계수의 유의성을 HAC "
            "표준오차 기준으로 재판단합니다 — 표본을 하나도 안 버리면서 더 정직한 p-value를 얻는 방식입니다."
        )

        if not _STATSMODELS_AVAILABLE:
            st.warning(
                "이 검정에는 `statsmodels` 패키지가 필요합니다. requirements.txt에 `statsmodels`를 "
                "추가하고 앱을 재배포해 주세요."
            )
        else:
            def _hac_regime_test(df, target_col, max_lag):
                """국면 더미변수 회귀 + HAC(Newey-West) 표준오차.
                기준(baseline)은 랭크가 가장 낮은(가장 안정적인) 국면이며,
                다른 국면 더미의 계수 = 기준 대비 평균 미래수익률 차이,
                HAC p-value = 그 차이의 자기상관 보정 유의성."""
                sub = df[["Regime", target_col]].dropna()
                regimes = sorted(sub["Regime"].unique())
                baseline = regimes[0]
                X = pd.DataFrame(index=sub.index)
                dummy_map = {}
                for r in regimes[1:]:
                    col = f"regime_{r}"
                    X[col] = (sub["Regime"] == r).astype(float)
                    dummy_map[r] = col
                X = sm.add_constant(X)
                y = sub[target_col].values
                model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": max_lag})

                rows = [{
                    "국면": state_map[baseline][0] + " (기준)",
                    "기준 대비 차이": np.nan,
                    "HAC 표준오차": np.nan,
                    "HAC p-value": np.nan,
                }]
                for r, col in dummy_map.items():
                    rows.append({
                        "국면": state_map[r][0],
                        "기준 대비 차이": model.params[col],
                        "HAC 표준오차": model.bse[col],
                        "HAC p-value": model.pvalues[col],
                    })
                return pd.DataFrame(rows).set_index("국면")

            try:
                # maxlags는 해당 수익률의 계산 기간(5일/20일)에서 -1을 적용
                # (Newey-West에서 권장되는 최소 랙 길이 = 겹치는 기간 - 1)
                hac5_df = _hac_regime_test(analyzed_df, "Fwd_Ret_5", max_lag=4)
                hac20_df = _hac_regime_test(analyzed_df, "Fwd_Ret_20", max_lag=19)

                st.markdown("**미래 5일 수익률 기준 (maxlags=4)**")
                st.dataframe(
                    hac5_df.style.format({
                        "기준 대비 차이": "{:.4%}", "HAC 표준오차": "{:.4%}", "HAC p-value": "{:.4f}",
                    }, na_rep="—").map(_highlight_sig, subset=["HAC p-value"])
                )

                st.markdown("**미래 20일 수익률 기준 (maxlags=19)**")
                st.dataframe(
                    hac20_df.style.format({
                        "기준 대비 차이": "{:.4%}", "HAC 표준오차": "{:.4%}", "HAC p-value": "{:.4f}",
                    }, na_rep="—").map(_highlight_sig, subset=["HAC p-value"])
                )

                st.caption(
                    "'기준 대비 차이'는 기준 국면(가장 안정적인 국면) 대비 해당 국면의 평균 미래수익률 "
                    "차이입니다. 'HAC p-value'가 이 차이의 통계적 유의성을 자기상관 보정 후 판단한 "
                    "값입니다. 위 다중 시작점 검정의 'p<0.05 비율'과 이 값을 함께 보고 최종 결론을 "
                    "내리세요 — 두 방식(표본 축소 vs 표준오차 보정)이 서로 다른 접근인데도 결론이 "
                    "일치한다면 신뢰도가 한층 높아지고, 갈린다면 그 국면 신호는 아직 불확실하다고 "
                    "보는 게 안전합니다."
                )
            except Exception as e:
                st.error(f"HAC 검정 중 오류가 발생했습니다: {e}")

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
