import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import json
import time
import os
import re
from datetime import datetime, timedelta
import FinanceDataReader as fdr
import urllib3

# Streamlit 클라우드 서버의 불필요한 SSL 보안 경고문 출력 강제 차단
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 1. 페이지 기본 설정
st.set_page_config(page_title="장중 퀀트 매트릭스", layout="wide")
st.title("📈 실전용 장중 매도/보유 판별 시스템 (커스텀 점수보드 탑재)")
st.caption(f"시스템 실시간 동기화: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (KST)")

try:
    APP_KEY = st.secrets["KOR_INVEST_APP_KEY"]
    APP_SECRET = st.secrets["KOR_INVEST_APP_SECRET"]
    URL_BASE = "https://openapi.koreainvestment.com:9443" 
except Exception as e:
    st.error("보안 키(Secrets)가 설정되지 않았습니다. Streamlit 대시보드 설정을 확인하세요.")
    st.stop()

# --- 도구 함수: HTML 커스텀 메트릭 렌더러 (화살표 방향 및 점수 완벽 제어) ---
def render_card(col, label, val_str, delta_text, arrow_type, color_type, score=None):
    # 화살표 방향 및 색상 통제 (상승=↑/빨강, 하락=↓/파랑, 중립=→/초록)
    arrow_map = {'up': '↑', 'down': '↓', 'flat': '→'}
    color_hex = {'red': '#ff4b4b', 'blue': '#0068c9', 'green': '#00e676', 'yellow': '#faca2b', 'white': '#d3d3d3'}
    
    a_char = arrow_map.get(arrow_type, '→')
    c_hex = color_hex.get(color_type, '#00e676')
    
    # 상단 Label 및 Value 출력 (기본 metric 활용하되 delta는 뺌)
    col.metric(label, val_str)
    
    # 옥토만경님의 요청: 초록색 세부 점수 렌더링 (score가 부여된 항목만)
    if score is not None:
        score_html = f"<div style='color: #00e676; font-size: 0.95rem; font-weight: bold; margin-top: 4px;'>↳ {score:+.2f}점</div>"
    else:
        score_html = ""
        
    # 화살표와 변화량을 합친 커스텀 HTML 주입
    delta_html = f"<div style='color: {c_hex}; font-size: 1rem; font-weight: bold; margin-top: -15px;'>{a_char} {delta_text}</div>"
    
    col.markdown(delta_html + score_html, unsafe_allow_html=True)

# 2. 한국투자증권 API 통신
def get_kis_token_shared(force_new=False):
    if not APP_KEY: return None
    token_file = "kis_token_shared.txt"
    if not force_new and os.path.exists(token_file) and (time.time() - os.path.getmtime(token_file) < 72000):
        with open(token_file, "r") as f:
            return f.read().strip()
    headers = {"content-type": "application/json"}
    body = {"grant_type": "client_credentials", "appkey": APP_KEY, "appsecret": APP_SECRET}
    try:
        res = requests.post(f"{URL_BASE}/oauth2/tokenP", headers=headers, data=json.dumps(body), timeout=5, verify=False)
        if res.status_code == 200:
            token = res.json().get("access_token")
            with open(token_file, "w") as f: f.write(token)
            return token
    except Exception:
        pass
    return None

class KISApi:
    def __init__(self):
        self.access_token = get_kis_token_shared()
        
    def safe_float(self, val):
        try:
            if val is None or str(val).strip() == "": return 0.0
            return float(str(val).replace(",", ""))
        except:
            return 0.0
        
    def get_current_price_and_vwap(self, stock_code, retry=True):
        if not self.access_token: return 0, 0, 0
        headers = {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self.access_token}",
            "appkey": APP_KEY,
            "appsecret": APP_SECRET,
            "tr_id": "FHKST01010100",
            "custtype": "P"
        }
        params = {"fid_cond_mrkt_div_code": "J", "fid_input_iscd": stock_code}
        try:
            res = requests.get(f"{URL_BASE}/uapi/domestic-stock/v1/quotations/inquire-price", headers=headers, params=params, verify=False, timeout=5)
            data = res.json()
            if data.get("rt_cd") == "0":
                out = data.get("output", {})
                stck_prpr = self.safe_float(out.get("stck_prpr"))
                prdy_ctrt = self.safe_float(out.get("prdy_ctrt")) 
                acml_vol = self.safe_float(out.get("acml_vol"))
                acml_tr_pbmn = self.safe_float(out.get("acml_tr_pbmn"))
                vwap = (acml_tr_pbmn / acml_vol) if acml_vol > 0 else stck_prpr
                return stck_prpr, vwap, prdy_ctrt
            elif retry: 
                self.access_token = get_kis_token_shared(force_new=True)
                return self.get_current_price_and_vwap(stock_code, retry=False)
        except Exception:
            pass
        return 0, 0, 0

    def get_investor_trend(self, stock_code):
        """3중 다중 방어망으로 수급 0주 출력 원천 차단"""
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0.0.0"}
        
        try:
            url_pc = f"https://finance.naver.com/item/frgn.naver?code={stock_code}"
            res_pc = requests.get(url_pc, headers=headers, timeout=5)
            res_pc.encoding = 'euc-kr'
            
            # 1. 장중 잠정치 우선 낚아채기
            prov_section = re.search(r'summary="외국인 기관 잠정치 추이"(.*?)</table', res_pc.text, re.DOTALL)
            if prov_section:
                matches = re.findall(r'<td class="tc"><span[^>]*>(\d{2}:\d{2})</span></td>\s*<td class="num"><span[^>]*>([+\-\d,]+)</span></td>\s*<td class="num"><span[^>]*>([+\-\d,]+)</span></td>', prov_section.group(1))
                for m in matches:
                    f_qty = float(m[1].replace(',', ''))
                    o_qty = float(m[2].replace(',', ''))
                    if f_qty != 0 or o_qty != 0:
                        return f_qty, o_qty, f"{m[0]} 잠정"
                        
            # 2. 잠정치가 없으면 일자별 확정치 역추적
            table_match = re.search(r'summary="외국인 기관 순매매 거래량"(.*?)</table', res_pc.text, re.DOTALL)
            if table_match:
                rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table_match.group(1), re.DOTALL)
                for r in rows:
                    cols = re.findall(r'<span class="tah[^>]*>(.*?)</span>', r)
                    if len(cols) >= 7:
                        date_str = cols[0].replace('.', '').strip()
                        if not date_str.isdigit() or len(date_str) < 8: continue
                        o_qty = float(cols[5].replace(',', ''))
                        f_qty = float(cols[6].replace(',', ''))
                        if f_qty != 0 or o_qty != 0:
                            return f_qty, o_qty, f"{date_str[4:6]}/{date_str[6:8]} 확정"
        except Exception:
            pass
            
        # 3. 최후의 보루: 네이버 모바일 JSON API
        try:
            url_m = f"https://m.stock.naver.com/api/stock/{stock_code}/investor/trend"
            res_m = requests.get(url_m, headers=headers, timeout=3)
            if res_m.status_code == 200:
                trend_list = res_m.json().get('result', {}).get('list', [])
                for row in trend_list:
                    f_qty = float(row.get('foreignerNetBuyVol', 0))
                    o_qty = float(row.get('organNetBuyVol', 0))
                    if f_qty != 0 or o_qty != 0:
                        bizdate = str(row.get('bizdate', ''))
                        formatted_date = f"{bizdate[4:6]}/{bizdate[6:8]}" if len(bizdate) == 8 else "최근"
                        return f_qty, o_qty, f"{formatted_date} API"
        except Exception:
            pass
            
        return 0.0, 0.0, "조회실패"

# 3. 데이터 패치 (글로벌 매크로 및 차트 지표)
@st.cache_data(ttl=60)
def fetch_global_macro_data():
    kospi_change = 0.0
    try:
        url = "https://finance.naver.com/sise/sise_index.naver?code=KOSPI"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0.0.0"}
        res = requests.get(url, headers=headers, timeout=5)
        res.encoding = 'euc-kr'
        match = re.search(r'id="change_value_and_rate"[^>]*>.*?([\+\-]?[\d\.]+)%', res.text, re.DOTALL)
        if match:
            kospi_change = float(match.group(1))
    except Exception:
        pass

    try:
        fx = yf.Ticker("KRW=X").history(period="5d")
        nq = yf.Ticker("NQ=F").history(period="5d")
        vix = yf.Ticker("^VIX").history(period="5d")
        
        fx_rate = fx['Close'].iloc[-1] if len(fx) > 0 else 0.0
        fx_change = ((fx_rate - fx['Close'].iloc[-2]) / fx['Close'].iloc[-2]) * 100 if len(fx) >= 2 else 0.0
        nq_rate = nq['Close'].iloc[-1] if len(nq) > 0 else 0.0
        nq_change = ((nq_rate - nq['Close'].iloc[-2]) / nq['Close'].iloc[-2]) * 100 if len(nq) >= 2 else 0.0
        vix_rate = vix['Close'].iloc[-1] if len(vix) > 0 else 0.0
            
        return fx_rate, fx_change, nq_rate, nq_change, vix_rate, kospi_change
    except Exception:
        return 0.0, 0.0, 0.0, 0.0, 0.0, kospi_change

@st.cache_data(ttl=3600)
def fetch_technical_indicators(stock_code):
    try:
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        df = fdr.DataReader(stock_code, start_date)
        if len(df) < 20: return 0.0, 50.0, 0.0, 0.0, 0.0
            
        ma20 = df['Close'].rolling(window=20).mean().iloc[-1]
        std20 = df['Close'].rolling(window=20).std().iloc[-1]
        upper_bb = ma20 + (std20 * 2)
        lower_bb = ma20 - (std20 * 2)
        
        delta = df['Close'].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        rs = up.ewm(com=13, adjust=False).mean() / down.ewm(com=13, adjust=False).mean()
        rsi_val = (100 - (100 / (1 + rs))).iloc[-1]
        
        df_recent = df.tail(60).copy()
        min_p, max_p = df_recent['Close'].min(), df_recent['Close'].max()
        if max_p == min_p: max_vol_price = min_p
        else:
            bin_size = (max_p - min_p) / 10
            bins = [min_p + i * bin_size for i in range(11)]
            df_recent['Bin_Idx'] = pd.cut(df_recent['Close'], bins=bins, labels=False, include_lowest=True)
            max_bin_idx = df_recent.groupby('Bin_Idx')['Volume'].sum().idxmax()
            max_vol_price = min_p + (max_bin_idx * bin_size) + (bin_size / 2)
            
        return ma20, rsi_val, max_vol_price, upper_bb, lower_bb
    except Exception:
        return 0.0, 50.0, 0.0, 0.0, 0.0

@st.cache_data(ttl=86400)
def load_krx_symbols():
    try:
        df = fdr.StockListing('KRX')
        if not df.empty: return dict(zip(df['Name'], df['Code']))
    except: pass
    try:
        df_k = fdr.StockListing('KOSPI')
        df_q = fdr.StockListing('KOSDAQ')
        df_tot = pd.concat([df_k, df_q])
        if not df_tot.empty: return dict(zip(df_tot['Name'], df_tot['Code']))
    except: pass
    
    return {"삼성전자": "005930", "SK하이닉스": "000660", "삼성전기": "009150", "에코프로": "086520", "현대차": "005380"}

# 4. 앱 화면 구성 및 종목 검색 UI 
st.sidebar.markdown("### 🔍 종목 자동완성 검색")

krx_dict = load_krx_symbols()
search_options = [f"{name} ({code})" for name, code in krx_dict.items()]
default_idx = next((i for i, opt in enumerate(search_options) if "005930" in opt), 0)

selected_stock = st.sidebar.selectbox("종목 선택 (클릭 후 타이핑)", options=search_options, index=default_idx)

stock_target = selected_stock.split("(")[-1].replace(")", "").strip()
stock_name = selected_stock.split("(")[0].strip()

# 5. 월가 수준 퀀트 알고리즘 실행
if st.sidebar.button(f"🚀 [{stock_name}] 데이터 분석 실행", type="primary"):
    with st.spinner(f"[{stock_name}({stock_target})] 실시간 데이터를 융합 분석 중입니다..."):
        
        kis_client = KISApi()
        
        cur_price, cur_vwap, stock_change = kis_client.get_current_price_and_vwap(stock_target)
        f_net, o_net, trend_date = kis_client.get_investor_trend(stock_target)
        
        fx_rate, fx_change, nq_rate, nq_change, vix_rate, kospi_change = fetch_global_macro_data()
        ma20, rsi14, vol_price, upper_bb, lower_bb = fetch_technical_indicators(stock_target)
        
        alpha = stock_change - kospi_change
        
        def calculate_institutional_score():
            if f_net == 0 and o_net == 0:
                s_f = 0.0
                s_orgn = 0.0
            else:
                s_f = 10.0 if f_net > 0 else -10.0
                s_orgn = 5.0 if o_net > 0 else -5.0                         
            
            s_e = max(-1.0, min(1.0, -fx_change / 0.5)) * 15.0          
            s_n = max(-1.0, min(1.0, nq_change / 1.0)) * 15.0           
            s_vix = 10.0 if vix_rate < 18 else (0.0 if vix_rate < 25 else -20.0) 
            
            macro_score = s_f + s_e + s_n + s_vix
            
            disparity = ((cur_price - cur_vwap) / cur_vwap * 100) if cur_vwap > 0 else 0
            s_v = max(-1.0, min(1.0, disparity / 2.0)) * 10.0           
            s_vol = 10.0 if cur_price >= vol_price else -10.0           
            s_ma = 10.0 if cur_price >= ma20 else -10.0                 
            s_alpha = 5.0 if alpha > 0 else -5.0                        
            
            s_bb = 0.0
            if cur_price >= upper_bb: s_bb = -10.0                      
            elif cur_price <= lower_bb: s_bb = 5.0                      
            
            s_rsi = 0.0
            if rsi14 >= 70: s_rsi = -5.0
            elif rsi14 <= 30: s_rsi = 5.0
            
            micro_score = s_v + s_vol + s_ma + s_orgn + s_alpha + s_bb + s_rsi
            
            final = macro_score + micro_score
            return max(-100.0, min(100.0, final)), disparity, macro_score, micro_score

        final_score, vwap_disp, m_score, t_score = calculate_institutional_score()

        st.subheader(f"🌐 거시 경제 및 시장 리스크 (Macro Factors)")
        m_cols = st.columns(4)
        
        # 1. 환율
        arr1, colr1 = ("up", "red") if fx_change > 0 else ("down", "blue") if fx_change < 0 else ("flat", "green")
        render_card(m_cols[0], "원/달러 환율", f"{fx_rate:,.1f}원", f"{fx_change:+.2f}%", arr1, colr1, s_e)
        
        # 2. 나스닥
        arr2, colr2 = ("up", "red") if nq_change > 0 else ("down", "blue") if nq_change < 0 else ("flat", "green")
        render_card(m_cols[1], "나스닥 선물", f"{nq_rate:,.1f}pt", f"{nq_change:+.2f}%", arr2, colr2, s_n)
        
        # 3. VIX
        arr3 = "flat" if vix_rate < 25 else "up"
        colr3 = "green" if vix_rate < 18 else "yellow" if vix_rate < 25 else "red"
        txt3 = "안정" if vix_rate < 18 else "경계" if vix_rate < 25 else "공포(발작)"
        render_card(m_cols[2], "VIX 공포 지수", f"{vix_rate:.2f}", txt3, arr3, colr3, s_vix)
        
        # 4. 코스피 (점수 부여 대상이 아니므로 점수는 표기 생략)
        arr4, colr4 = ("up", "red") if kospi_change > 0 else ("down", "blue") if kospi_change < 0 else ("flat", "green")
        render_card(m_cols[3], "코스피 지수", f"{kospi_change:+.2f}% (실시간)", f"{kospi_change:+.2f}%", arr4, colr4, None)

        st.markdown("---")
        st.subheader(f"🔍 [{stock_name}({stock_target})] 자금 이탈입 및 기술적 지표 (Micro Factors)")
        t_cols1 = st.columns(4)
        
        # 5. 현재가 (VWAP 이격도 설명 추가)
        arr5, colr5 = ("up", "red") if stock_change > 0 else ("down", "blue") if stock_change < 0 else ("flat", "green")
        render_card(t_cols1[0], "대상 종목 현재가", f"{cur_price:,.0f}원", f"{stock_change:+.2f}% (VWAP 이격: {vwap_disp:+.2f}%)", arr5, colr5, s_v)
        
        # 6. 상대강도(알파)
        arr6, colr6 = ("up", "red") if alpha > 0 else ("down", "blue") if alpha < 0 else ("flat", "green")
        txt6 = f"{alpha:+.2f}% (주도주)" if alpha > 0 else f"{alpha:+.2f}% (시장 하회)" if alpha < 0 else f"0.00% (동조)"
        render_card(t_cols1[1], "코스피 대비 상대강도(α)", f"{alpha:+.2f}%", txt6, arr6, colr6, s_alpha)
        
        # 7. 외국인
        if f_net > 0: arr7, colr7, txt7 = "up", "red", f"+{f_net:,.0f}주"
        elif f_net < 0: arr7, colr7, txt7 = "down", "blue", f"{f_net:,.0f}주"
        else: arr7, colr7, txt7 = "flat", "white", "0주 (판단 제외)"
        render_card(t_cols1[2], f"외국인 순매매 ({trend_date})", f"{f_net:,.0f}주", txt7, arr7, colr7, s_f)
        
        # 8. 기관
        if o_net > 0: arr8, colr8, txt8 = "up", "red", f"+{o_net:,.0f}주"
        elif o_net < 0: arr8, colr8, txt8 = "down", "blue", f"{o_net:,.0f}주"
        else: arr8, colr8, txt8 = "flat", "white", "0주 (판단 제외)"
        render_card(t_cols1[3], f"기관 순매매 ({trend_date})", f"{o_net:,.0f}주", txt8, arr8, colr8, s_orgn)

        t_cols2 = st.columns(4)
        
        # 9. 매물대
        arr9, colr9 = ("up", "red") if cur_price >= vol_price else ("down", "blue")
        txt9 = "지지층 확보" if cur_price >= vol_price else "악성 저항대"
        render_card(t_cols2[0], "최대 매물대 (60일)", f"{vol_price:,.0f}원", txt9, arr9, colr9, s_vol)
        
        # 10. 이동평균선
        arr10, colr10 = ("up", "red") if cur_price >= ma20 else ("down", "blue")
        txt10 = "상승 추세" if cur_price >= ma20 else "하락 추세"
        render_card(t_cols2[1], "20일 이동평균선", f"{ma20:,.0f}원", txt10, arr10, colr10, s_ma)
        
        # 11. 볼린저 밴드
        if cur_price >= upper_bb: arr11, colr11, txt11 = "up", "red", "상단 터치(과열)"
        elif cur_price <= lower_bb: arr11, colr11, txt11 = "down", "blue", "하단(반등대기)"
        else: arr11, colr11, txt11 = "flat", "green", "밴드 내 안정"
        render_card(t_cols2[2], "볼린저 밴드 한계치", f"{upper_bb:,.0f}원 (상단)", txt11, arr11, colr11, s_bb)
        
        # 12. RSI
        if rsi14 >= 70: arr12, colr12, txt12 = "up", "red", "과매수(조정주의)"
        elif rsi14 <= 30: arr12, colr12, txt12 = "down", "blue", "과매도(반등기대)"
        else: arr12, colr12, txt12 = "flat", "green", "중립"
        render_card(t_cols2[3], "RSI (14일)", f"{rsi14:.1f}", txt12, arr12, colr12, s_rsi)

        st.markdown("---")
        st.subheader("🎯 기관급 퀀트 알고리즘 최종 판단")
        
        no_supply = (f_net == 0 and o_net == 0)

        if final_score >= 40:
            desc = "매물대 돌파 및 매크로 환경이 매우 강력한 상승을 지시하고 있습니다. (수급 데이터 판단 제외)" if no_supply else "기관 쌍끌이 수급, 매물대 돌파, 매크로 환경이 모두 완벽한 삼위일체를 이루고 있습니다."
            st.success(f"🔴 [강력 보유/매수] 종합 지수: {final_score:+.1f}점 (매크로 {m_score:+.1f}점 / 마이크로 {t_score:+.1f}점)\n\n{desc}")
        elif 0 <= final_score < 40:
            desc = "추세는 양호하나 볼린저 밴드 과열 또는 거시적 저항 여부를 주시하세요. (수급 데이터 판단 제외)" if no_supply else "추세는 양호하나 볼린저 밴드 과열 또는 거시적 저항 여부를 주시하세요."
            st.info(f"🟢 [보유 유지] 종합 지수: {final_score:+.1f}점 (매크로 {m_score:+.1f}점 / 마이크로 {t_score:+.1f}점)\n\n{desc}")
        elif -30 <= final_score < 0:
            desc = "시장 주도력을 잃고 있습니다. 리스크 관리를 시작하십시오. (수급 데이터 판단 제외)" if no_supply else "시장 주도력을 잃거나 자금이 이탈 중입니다. 리스크 관리를 시작하십시오."
            st.warning(f"🟡 [관망 주의] 종합 지수: {final_score:+.1f}점 (매크로 {m_score:+.1f}점 / 마이크로 {t_score:+.1f}점)\n\n{desc}")
        elif -60 <= final_score < -30:
            desc = "매물대 저항이 강합니다. 철저히 기계적으로 분할 매도하여 비중을 줄이십시오. (수급 데이터 판단 제외)" if no_supply else "수급 이탈과 매물대 저항이 겹쳤습니다. 철저히 기계적으로 분할 매도하여 비중을 줄이십시오."
            st.error(f"🔵 [비중 축소] 종합 지수: {final_score:+.1f}점 (매크로 {m_score:+.1f}점 / 마이크로 {t_score:+.1f}점)\n\n{desc}")
        else:
            desc = "VIX 공포 상승 등 시장 환경과 차트가 무너진 상태입니다. 전량 현금화를 권장합니다. (수급 데이터 판단 제외)" if no_supply else "VIX 공포 상승 등 시장 환경과 차트, 수급이 모두 무너진 상태입니다. 전량 현금화를 권장합니다."
            st.error(f"🔵 [강력 매도] 종합 지수: {final_score:+.1f}점 (매크로 {m_score:+.1f}점 / 마이크로 {t_score:+.1f}점)\n\n{desc}")
