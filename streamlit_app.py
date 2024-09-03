# imports
import yfinance as yf
import streamlit as st
import datetime 
import pandas as pd
import cufflinks as cf
from plotly.offline import iplot
from plotly.subplots import make_subplots
#from streamlit.cache import cache_data, cache_resource

## set offline mode for cufflinks
cf.go_offline()

# data functions
@st.cache_data
def get_sp500_components():
    df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    df = df[0]
    tickers = df["Symbol"].to_list()
    tickers_companies_dict = dict(
        zip(df["Symbol"], df["Security"])
    )
    return tickers, tickers_companies_dict

@st.cache_data
def load_data(symbol, start, end):
    return yf.download(symbol, start, end)

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv().encode("utf-8")

# sidebar

## inputs for downloading data
st.sidebar.header("Stock Parameters")

available_tickers, tickers_companies_dict = get_sp500_components()

ticker = st.sidebar.selectbox(
    "Ticker", 
    available_tickers, 
    format_func=tickers_companies_dict.get
)
start_date = st.sidebar.date_input(
    "Start date", 
    datetime.date(2019, 1, 1)
)
end_date = st.sidebar.date_input(
    "End date", 
    datetime.date.today()
)

if start_date > end_date:
    st.sidebar.error("The end date must fall after the start date")

## inputs for technical analysis
st.sidebar.header("Technical Analysis Parameters")

volume_flag = st.sidebar.checkbox(label="Add volume")

exp_sma = st.sidebar.expander("SMA")
sma_flag = exp_sma.checkbox(label="Add SMA")
sma_periods= exp_sma.number_input(
    label="SMA Periods", 
    min_value=1, 
    max_value=50, 
    value=20, 
    step=1
)

exp_bb = st.sidebar.expander("Bollinger Bands")
bb_flag = exp_bb.checkbox(label="Add Bollinger Bands")
bb_periods= exp_bb.number_input(label="BB Periods", 
                                min_value=1, max_value=50, 
                                value=20, step=1)
bb_std= exp_bb.number_input(label="# of standard deviations", 
                            min_value=1, max_value=4, 
                            value=2, step=1)

exp_rsi = st.sidebar.expander("Relative Strength Index")
rsi_flag = exp_rsi.checkbox(label="Add RSI")
rsi_periods= exp_rsi.number_input(
    label="RSI Periods", 
    min_value=1, 
    max_value=50, 
    value=20, 
    step=1
)
rsi_upper= exp_rsi.number_input(label="RSI Upper", 
                                min_value=50, 
                                max_value=90, value=70, 
                                step=1)
rsi_lower= exp_rsi.number_input(label="RSI Lower", 
                                min_value=10, 
                                max_value=50, value=30, 
                                step=1)

# 메인 부분
st.title("티커 기술적 분석 웹 서비스")
st.write("""
### User manual
- S&P 지수의 구성 요소인 모든 회사를 선택할 수 있습니다.
- 관심 있는 기간을 선택할 수 있습니다.
- 선택한 데이터를 CSV 파일로 다운로드할 수 있습니다.
- 다음 기술적 지표를 플롯에 추가할 수 있습니다: 단순 이동 평균, 볼린저 밴드, 상대 강도 지수
- 지표의 다양한 매개변수를 실험해 볼 수 있습니다.
""")

# 차트 생성
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])

# 캔들스틱 차트
fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='OHLC'), row=1, col=1)

if sma_flag:
    df = add_sma(df, sma_periods)
    fig.add_trace(go.Scatter(x=df.index, y=df[f'SMA_{sma_periods}'], name=f'SMA {sma_periods}', line=dict(color='orange', width=1)), row=1, col=1)

if bb_flag:
    df = add_bollinger_bands(df, bb_periods, bb_std)
    fig.add_trace(go.Scatter(x=df.index, y=df[f'BB_upper_{bb_periods}'], name=f'BB Upper', line=dict(color='lightgrey', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df[f'BB_middle_{bb_periods}'], name=f'BB Middle', line=dict(color='grey', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df[f'BB_lower_{bb_periods}'], name=f'BB Lower', line=dict(color='lightgrey', width=1)), row=1, col=1)

if volume_flag:
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='lightblue'), row=2, col=1)

if rsi_flag:
    df = add_rsi(df, rsi_periods)
    fig.add_trace(go.Scatter(x=df.index, y=df[f'RSI_{rsi_periods}'], name=f'RSI {rsi_periods}', line=dict(color='purple', width=1)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

fig.update_layout(
    title=f"{tickers_companies_dict[ticker]}'s stock price",
    yaxis_title="Price",
    xaxis_rangeslider_visible=False,
    height=800
)

st.plotly_chart(fig, use_container_width=True)

# Auto ARIMA 분석 섹션
# ARIMA 분석 섹션
st.header("ARIMA 분석")
if st.button("ARIMA 분석 수행"):
    with st.spinner("ARIMA 분석 중..."):
        forecast, summary = perform_arima_analysis(df['Close'])
        
        # 모델 요약
        st.subheader("ARIMA 모델 요약")
        st.text(summary)
        
        # 예측 및 시각화
        index_of_fc = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')

        # 결과 시각화
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='실제 가격'))
        fig.add_trace(go.Scatter(x=index_of_fc, y=forecast, mode='lines', name='예측', line=dict(color='red')))
        
        fig.update_layout(title='주가와 ARIMA 예측',
                          xaxis_title='날짜',
                          yaxis_title='가격',
                          height=600)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 트렌드 존재 여부 판단
        trend_diff = np.diff(forecast)
        trend_direction = np.mean(trend_diff)
        
        if abs(trend_direction) > 0.01:  # 임계값 설정 (필요에 따라 조정 가능)
            trend_message = "상승" if trend_direction > 0 else "하락"
            st.success(f"분석 결과, 향후 30일 동안 {trend_message} 트렌드가 예상됩니다.")
        else:
            st.info("분석 결과, 향후 30일 동안 뚜렷한 트렌드가 없을 것으로 예상됩니다.")
