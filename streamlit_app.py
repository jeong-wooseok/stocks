import plotly.graph_objects as go
import yfinance as yf
import streamlit as st
import datetime 
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
from statsmodels.tsa.arima.model import ARIMA

# data functions
@st.cache_data
def get_sp500_components():
    df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    df = df[0]
    tickers = df["Symbol"].to_list()
    tickers_companies_dict = dict(zip(df["Symbol"], df["Security"]))
    return tickers, tickers_companies_dict

@st.cache_data
def load_data(symbol, start, end):
    return yf.download(symbol, start, end)

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv().encode("utf-8")

# Technical indicators
def add_sma(df, period):
    df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
    return df

def add_bollinger_bands(df, period, std_dev):
    df[f'BB_middle_{period}'] = df['Close'].rolling(window=period).mean()
    df[f'BB_upper_{period}'] = df[f'BB_middle_{period}'] + (df['Close'].rolling(window=period).std() * std_dev)
    df[f'BB_lower_{period}'] = df[f'BB_middle_{period}'] - (df['Close'].rolling(window=period).std() * std_dev)
    return df

def add_rsi(df, period):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
    return df

# 시계열 분해 함수
def perform_time_series_decomposition(data):
    decomposition = seasonal_decompose(data, model='additive', period=30)
    
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    
    return trend, seasonal, residual

# ARIMA model function
def perform_arima_analysis(data):
    model = ARIMA(data, order=(1,1,1))
    results = model.fit()
    
    # Model summary
    summary = str(results.summary())
    
    # Forecast
    forecast = results.forecast(steps=30)
    
    return forecast, summary

# Sidebar
st.sidebar.header("Stock Parameters")
available_tickers, tickers_companies_dict = get_sp500_components()
ticker = st.sidebar.selectbox("Ticker", available_tickers, format_func=tickers_companies_dict.get)
start_date = st.sidebar.date_input("Start date", datetime.date(2019, 1, 1))
end_date = st.sidebar.date_input("End date", datetime.date.today())

if start_date > end_date:
    st.sidebar.error("The end date must fall after the start date")

# Technical Analysis Parameters
st.sidebar.header("Technical Analysis Parameters")
volume_flag = st.sidebar.checkbox(label="Add volume")
sma_flag = st.sidebar.checkbox(label="Add SMA")
sma_periods = st.sidebar.number_input("SMA Periods", min_value=1, max_value=50, value=20, step=1)
bb_flag = st.sidebar.checkbox(label="Add Bollinger Bands")
bb_periods = st.sidebar.number_input("BB Periods", min_value=1, max_value=50, value=20, step=1)
bb_std = st.sidebar.number_input("# of standard deviations", min_value=1, max_value=4, value=2, step=1)
rsi_flag = st.sidebar.checkbox(label="Add RSI")
rsi_periods = st.sidebar.number_input("RSI Periods", min_value=1, max_value=50, value=14, step=1)

# Main body
st.title("티커 기술적 분석 웹 서비스")
st.write("""
### User manual
- S&P 지수의 구성 요소인 모든 회사를 선택할 수 있습니다.
- 관심 있는 기간을 선택할 수 있습니다.
- 선택한 데이터를 CSV 파일로 다운로드할 수 있습니다.
- 다음 기술적 지표를 플롯에 추가할 수 있습니다: 단순 이동 평균, 볼린저 밴드, 상대 강도 지수
- 지표의 다양한 매개변수를 실험해 볼 수 있습니다.
""")

# Load data
df = load_data(ticker, start_date, end_date)

# Data preview
data_exp = st.expander("Preview data")
available_cols = df.columns.tolist()
columns_to_show = data_exp.multiselect("Columns", available_cols, default=available_cols)
data_exp.dataframe(df[columns_to_show])

csv_file = convert_df_to_csv(df[columns_to_show])
data_exp.download_button(
    label="Download selected as CSV",
    data=csv_file,
    file_name=f"{ticker}_stock_prices.csv",
    mime="text/csv",
)

# Chart creation
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])

# Candlestick chart
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


# 시계열 분해 섹션
st.header("시계열 분해 분석")
if st.button("시계열 분해 수행"):
    with st.spinner("시계열 분해 중..."):
        trend, seasonal, residual = perform_time_series_decomposition(df['Close'])
        
        # 결과 시각화
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            subplot_titles=("원본 데이터", "추세", "계절성", "잔차"))
        
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='원본 데이터'), row=1, col=1)
        fig.add_trace(go.Scatter(x=trend.index, y=trend, mode='lines', name='추세'), row=2, col=1)
        fig.add_trace(go.Scatter(x=seasonal.index, y=seasonal, mode='lines', name='계절성'), row=3, col=1)
        fig.add_trace(go.Scatter(x=residual.index, y=residual, mode='lines', name='잔차'), row=4, col=1)
        
        fig.update_layout(height=900, title_text="시계열 분해 결과")
        st.plotly_chart(fig, use_container_width=True)
        
        # 추세 분석
        overall_trend = trend.iloc[-1] - trend.iloc[0]
        if overall_trend > 0:
            st.success(f"분석 결과, 선택한 기간 동안 전반적인 상승 추세가 관찰됩니다. (총 변화: {overall_trend:.2f})")
        elif overall_trend < 0:
            st.error(f"분석 결과, 선택한 기간 동안 전반적인 하락 추세가 관찰됩니다. (총 변화: {overall_trend:.2f})")
        else:
            st.info("분석 결과, 선택한 기간 동안 뚜렷한 추세가 관찰되지 않습니다.")
        
        # 계절성 분석
        max_seasonality = seasonal.max()
        min_seasonality = seasonal.min()
        st.info(f"계절성 변동 범위: {min_seasonality:.2f} ~ {max_seasonality:.2f}")
        
        # 잔차 분석
        residual_std = residual.std()
        st.info(f"잔차의 표준편차: {residual_std:.2f}")
