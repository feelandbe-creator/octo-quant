import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import pytz

# --- 1. 기본 설정 ---
st.set_page_config(page_title="Post-Market Accuracy Tracker", layout="wide")

kst_tz = pytz.timezone('Asia/Seoul')
now_utc = datetime.utcnow().replace(tzinfo=pytz.utc)
kst_time = now_utc.astimezone(kst_tz).strftime('%Y-%m-%d %H:%M:%S')

st.title("🎯 Post-Market Model Accuracy Tracker")
st.markdown("정규장 마감 후, 프리마켓 예측 데이터와 실제 주가 변동의 괴리율(Tracking Error)을 추적하고 모델을 교정합니다.")
st.caption(f"최종 업데이트 (KST): {kst_time}")
st.divider()

# --- 2. 검증 데이터 목업 (실제 DB 또는 로그 파일 연동 필요) ---
def get_evaluation_data():
    # 예측 변동폭 중간값과 실제 수익률(시가 대비 종가) 비교
    return pd.DataFrame({
        '티커': ['JOBY', 'VRT', 'CRWD', 'TSLA', 'DG', 'F'],
        '섹터': ['UAM', 'AI인프라', '보안', '자율주행', '유통', '자동차'],
        '예측 방향': ['Long', 'Long', 'Long', 'Long', 'Short', 'Short'],
        '예측값(%)': [5.5, 5.5, 4.25, 2.75, -4.75, -3.25], # 예측 변동폭 중간값
        '실제값(%)': [6.8, 5.0, 1.2, -1.5, -5.2, -2.8],
        '예측_NLP점수': [0.92, 0.88, 0.81, 0.60, -0.85, -0.70]
    })

df = get_evaluation_data()

# 오차 연산 로직
df['오차율(%)'] = df['실제값(%)'] - df['예측값(%)']
df['절대오차(|%|)'] = df['오차율(%)'].abs()
df['적중 여부'] = np.where(
    ((df['예측 방향'] == 'Long') & (df['실제값(%)'] > 0)) | 
    ((df['예측 방향'] == 'Short') & (df['실제값(%)'] < 0)), 
    'Hit 🎯', 'Miss ❌'
)

# --- 3. 성과 요약 메트릭 ---
hit_rate = (len(df[df['적중 여부'] == 'Hit 🎯']) / len(df)) * 100
mae = df['절대오차(|%|)'].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("전체 모델 적중률 (Hit Rate)", f"{hit_rate:.1f}%", f"{hit_rate - 75:.1f}% vs 목표치(75%)")
col2.metric("평균 절대 오차 (MAE)", f"{mae:.2f}%", "-0.15% 개선됨", delta_color="inverse")
col3.metric("롱 포지션(Long) 최고 적중", df.loc[df['실제값(%)'].idxmax()]['티커'])
col4.metric("숏 포지션(Short) 최고 적중", df.loc[df['실제값(%)'].idxmin()]['티커'])

st.divider()

# --- 4. 시각화: 예측 vs 실제 산점도 (Scatter Plot) ---
st.header("1. 예측값 vs 실제 변동률 괴리 분석")

fig = px.scatter(
    df, x='예측값(%)', y='실제값(%)', color='적중 여부', hover_data=['티커', '섹터'],
    color_discrete_map={'Hit 🎯': '#00cc96', 'Miss ❌': '#ef553b'},
    title="완벽한 예측은 대각선(y=x) 상에 위치합니다."
)
# 완벽한 일치 기준선 (y=x)
fig.add_shape(type="line", x0=-8, y0=-8, x1=8, y1=8, line=dict(color="gray", dash="dash"))
fig.update_layout(height=500, paper_bgcolor="rgba(0,0,0,0)", font={'color': "#c9d1d9"})
st.plotly_chart(fig, use_container_width=True)

# --- 5. 상세 데이터 테이블 및 모델 교정 제안 ---
st.header("2. 개별 종목 검증 결과 및 피드백")
st.dataframe(
    df[['티커', '섹터', '예측 방향', '예측값(%)', '실제값(%)', '오차율(%)', '적중 여부']],
    use_container_width=True
)

st.subheader("💡 퀀트 모델 교정 제안 (Auto-Calibration Insight)")
missed_df = df[df['적중 여부'] == 'Miss ❌']

if not missed_df.empty:
    for _, row in missed_df.iterrows():
        st.warning(
            f"**{row['티커']} 오차 경고:** 예측값({row['예측값(%)']}%) 대비 실제값({row['실제값(%)']}%) 역진행 발생. "
            f"개장 전 NLP 점수({row['예측_NLP점수']})가 과대평가되었거나, 장 초반 이동평균선 저항을 돌파하지 못한 기술적 한계로 추정됩니다. "
            f"다음 거래일 해당 섹터의 과거 패턴 승률 가중치를 10% 하향 조정하는 것을 권장합니다."
        )
else:
    st.success("모든 종목의 방향성이 적중했습니다. 현재 가중치 세팅을 유지하십시오.")
