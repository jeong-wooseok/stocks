import plotly.graph_objects as go
import yfinance as yf
import streamlit as st
import datetime 
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
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

# 차트 색상 팔레트 정의
color_palette = {
    'background': '#F0F2F6',
    'text': '#262730',
    'grid': '#B0BEC5',
    'candlestick_increasing': '#26A69A',
    'candlestick_decreasing': '#EF5350',
    'volume': '#90CAF9',
    'sma': '#FB8C00',
    'bb_upper': '#7E57C2',
    'bb_middle': '#5E35B1',
    'bb_lower': '#7E57C2',
    'rsi': '#F06292',
    'log_data': '#1E88E5',
    'diff_data': '#43A047',
    'trend': '#FFA000',
    'seasonal': '#5E35B1',
    'residual': '#E53935',
    'forecast': '#FF6F00'
}

# ARIMA 모델 함수
def perform_arima_analysis(data):
    model = ARIMA(data, order=(1,1,1))
    results = model.fit()
    
    # 모델 요약
    summary = str(results.summary())
    
    # 예측
    forecast = results.forecast(steps=30)
    
    # 트렌드 판단
    last_value = data.iloc[-1]
    forecast_end = forecast.iloc[-1]
    percent_change = (forecast_end - last_value) / last_value * 100
    
    if percent_change > 5:
        trend = "상승"
    elif percent_change < -5:
        trend = "하락"
    else:
        trend = "횡보"
    
    return forecast, summary, trend, percent_change

# 시계열 분해 함수
def perform_time_series_decomposition(data):
    # 결측값 처리
    data = data.dropna()
    
    # 인덱스가 DatetimeIndex인지 확인하고 아니면 변환
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    
    # 일별 데이터로 리샘플링 (필요한 경우)
    data = data.resample('D').mean()
    
    # 결측값이 있으면 보간
    data = data.interpolate()
    
    # 로그 변환 (선택적)
    log_data = np.log(data)
    
    # 차분
    diff_data = log_data.diff().dropna()
    
    # 원본 데이터 분해
    decomposition = seasonal_decompose(log_data, model='additive', period=30)
    
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    
    return log_data, diff_data, trend, seasonal, residual

# 추세 분석 함수
def analyze_trend(trend):
    overall_trend = trend.iloc[-1] - trend.iloc[0]
    if overall_trend > 0:
        return f"상승 추세가 관찰됩니다. (총 변화: {np.exp(overall_trend) - 1:.2%})"
    elif overall_trend < 0:
        return f"하락 추세가 관찰됩니다. (총 변화: {np.exp(overall_trend) - 1:.2%})"
    else:
        return "뚜렷한 추세가 관찰되지 않습니다."

# ADF 테스트 함수
def perform_adf_test(data):
    result = adfuller(data.dropna())
    return f'ADF 통계량: {result[0]:.4f}, p-value: {result[1]:.4f}'

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

# 차트 생성 함수 추가
def create_stock_chart(df, volume_flag, sma_flag, sma_periods, bb_flag, bb_periods, bb_std, rsi_flag, rsi_periods):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])

    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='OHLC', increasing_line_color=color_palette['candlestick_increasing'],
        decreasing_line_color=color_palette['candlestick_decreasing']
    ), row=1, col=1)

    if sma_flag:
        df = add_sma(df, sma_periods)
        fig.add_trace(go.Scatter(x=df.index, y=df[f'SMA_{sma_periods}'], name=f'SMA {sma_periods}', 
                                 line=dict(color=color_palette['sma'], width=1)), row=1, col=1)

    if bb_flag:
        df = add_bollinger_bands(df, bb_periods, bb_std)
        fig.add_trace(go.Scatter(x=df.index, y=df[f'BB_upper_{bb_periods}'], name=f'BB Upper', 
                                 line=dict(color=color_palette['bb_upper'], width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df[f'BB_middle_{bb_periods}'], name=f'BB Middle', 
                                 line=dict(color=color_palette['bb_middle'], width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df[f'BB_lower_{bb_periods}'], name=f'BB Lower', 
                                 line=dict(color=color_palette['bb_lower'], width=1)), row=1, col=1)

    if volume_flag:
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', 
                             marker_color=color_palette['volume']), row=2, col=1)

    if rsi_flag:
        df = add_rsi(df, rsi_periods)
        fig.add_trace(go.Scatter(x=df.index, y=df[f'RSI_{rsi_periods}'], name=f'RSI {rsi_periods}', 
                                 line=dict(color=color_palette['rsi'], width=1)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    fig.update_layout(
        title=f"{tickers_companies_dict[ticker]}'s stock price",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=800,
        plot_bgcolor=color_palette['background'],
        paper_bgcolor=color_palette['background'],
        font_color=color_palette['text'],
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=color_palette['grid'])
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=color_palette['grid'])

    return fig

# 시계열 분해 차트 생성 함수 추가
def create_decomposition_chart(log_data, diff_data, trend, seasonal, residual):
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=("원본 데이터 (로그 스케일)", "차분된 데이터", "추세", "계절성", "잔차"))
    
    fig.add_trace(go.Scatter(x=log_data.index, y=log_data, mode='lines', name='원본 데이터 (로그)', 
                             line=dict(color=color_palette['log_data'])), row=1, col=1)
    fig.add_trace(go.Scatter(x=diff_data.index, y=diff_data, mode='lines', name='차분된 데이터', 
                             line=dict(color=color_palette['diff_data'])), row=2, col=1)
    fig.add_trace(go.Scatter(x=trend.index, y=trend, mode='lines', name='추세', 
                             line=dict(color=color_palette['trend'])), row=3, col=1)
    fig.add_trace(go.Scatter(x=seasonal.index, y=seasonal, mode='lines', name='계절성', 
                             line=dict(color=color_palette['seasonal'])), row=4, col=1)
    fig.add_trace(go.Scatter(x=residual.index, y=residual, mode='lines', name='잔차', 
                             line=dict(color=color_palette['residual'])), row=5, col=1)
    
    fig.update_layout(
        height=1200, 
        title_text="시계열 분해 결과",
        plot_bgcolor=color_palette['background'],
        paper_bgcolor=color_palette['background'],
        font_color=color_palette['text'],
        hovermode='x unified'
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=color_palette['grid'])
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=color_palette['grid'])

    return fig

# 차트 생성
st.plotly_chart(create_stock_chart(df, volume_flag, sma_flag, sma_periods, bb_flag, bb_periods, bb_std, rsi_flag, rsi_periods), use_container_width=True)

# 시계열 분해 섹션
st.header("시계열 분해 분석")
if st.button("시계열 분해 수행"):
    with st.spinner("시계열 분해 중..."):
        log_data, diff_data, trend, seasonal, residual = perform_time_series_decomposition(df['Close'])
        
        st.plotly_chart(create_decomposition_chart(log_data, diff_data, trend, seasonal, residual), use_container_width=True)
        
        # 추세 분석
        st.subheader("추세 분석")
        forecast, summary, arima_trend, percent_change = perform_arima_analysis(df['Close'])
        
        if arima_trend == "상승":
            st.success(f"ARIMA 분석 결과, 향후 30일 동안 상승 추세가 예상됩니다. (예상 변화: {percent_change:.2f}%)")
        elif arima_trend == "하락":
            st.error(f"ARIMA 분석 결과, 향후 30일 동안 하락 추세가 예상됩니다. (예상 변화: {percent_change:.2f}%)")
        else:
            st.info(f"ARIMA 분석 결과, 향후 30일 동안 뚜렷한 추세가 없을 것으로 예상됩니다. (예상 변화: {percent_change:.2f}%)")
        
        # ARIMA 예측 결과 시각화
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='실제 가격', line=dict(color=color_palette['log_data'])))
        fig.add_trace(go.Scatter(x=pd.date_range(start=df.index[-1], periods=31, freq='D')[1:], 
                                 y=forecast, mode='lines', name='ARIMA 예측', line=dict(color=color_palette['forecast'])))
        
        fig.update_layout(title='ARIMA 모델 예측 결과',
                          xaxis_title='날짜',
                          yaxis_title='가격',
                          height=500,
                          plot_bgcolor=color_palette['background'],
                          paper_bgcolor=color_palette['background'],
                          font_color=color_palette['text'],
                          hovermode='x unified')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 계절성 분석
        st.subheader("계절성 분석")
        max_seasonality = seasonal.max()
        min_seasonality = seasonal.min()
        st.info(f"계절성 변동 범위: {np.exp(min_seasonality) - 1:.2%} ~ {np.exp(max_seasonality) - 1:.2%}")
        
        # 잔차 분석
        st.subheader("잔차 분석")
        residual_std = residual.std()
        st.info(f"잔차의 표준편차: {residual_std:.4f}")
        
        # 정상성 검정
        st.subheader("정상성 검정 (ADF 테스트)")
        st.info("원본 데이터: " + perform_adf_test(log_data))
        st.info("차분된 데이터: " + perform_adf_test(diff_data))
