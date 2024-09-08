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

# 차트 색상 팔레트
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

# 데이터 함수
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

# 기술적 지표
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

# ARIMA 분석
def perform_arima_analysis(data):
    try:
        if 'Close' not in data.columns:
            return None, None, "'Close' 열이 데이터에 없습니다.", None, None
        
        close_data = data['Close'].dropna()
        
        if len(close_data) < 30:
            return None, None, "데이터가 충분하지 않습니다. 최소 30일 이상의 데이터가 필요합니다.", None, None
        
        log_data = np.log(close_data)
        
        model = ARIMA(log_data, order=(1,1,1))
        results = model.fit()
        
        summary = str(results.summary())
        
        forecast_log_30 = results.forecast(steps=30)
        forecast_log_7 = forecast_log_30[:7]
        
        forecast_30 = np.exp(forecast_log_30)
        forecast_7 = np.exp(forecast_log_7)
        
        last_value = close_data.iloc[-1]
        forecast_end = forecast_30.iloc[-1]
        
        percent_change = ((forecast_end - last_value) / last_value) * 100
        
        if percent_change > 5:
            trend = "상승"
        elif percent_change < -5:
            trend = "하락"
        else:
            trend = "횡보"
        
        return forecast_30, forecast_7, summary, trend, percent_change
    
    except Exception as e:
        error_msg = f"ARIMA 분석 중 오류가 발생했습니다: {str(e)}"
        st.error(error_msg)
        return None, None, error_msg, None, None

# 시계열 분해
def perform_time_series_decomposition(data):
    data = data.dropna()
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    data = data.resample('D').mean().interpolate()
    
    log_data = np.log(data)
    diff_data = log_data.diff().dropna()
    
    decomposition = seasonal_decompose(log_data, model='additive', period=30)
    
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    
    return log_data, diff_data, trend, seasonal, residual

# ADF 테스트
def perform_adf_test(data):
    result = adfuller(data.dropna())
    return f'ADF 통계량: {result[0]:.4f}, p-value: {result[1]:.4f}'

# 차트 생성
def create_stock_chart(df, volume_flag, sma_flag, sma_periods, bb_flag, bb_periods, bb_std, rsi_flag, rsi_periods):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])

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

# 시계열 분해 차트
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

# 메인 함수
def main():
    st.title("티커 기술적 분석 웹 서비스")

    # 사이드바
    st.sidebar.header("주식 매개변수")
    available_tickers, tickers_companies_dict = get_sp500_components()
    ticker = st.sidebar.selectbox("티커", available_tickers, format_func=tickers_companies_dict.get)
    start_date = st.sidebar.date_input("시작일", datetime.date(2019, 1, 1))
    end_date = st.sidebar.date_input("종료일", datetime.date.today())

    if start_date > end_date:
        st.sidebar.error("종료일은 시작일 이후여야 합니다.")

    st.sidebar.header("기술적 분석 매개변수")
    volume_flag = st.sidebar.checkbox(label="거래량 추가")
    sma_flag = st.sidebar.checkbox(label="SMA 추가")
    sma_periods = st.sidebar.number_input("SMA 기간", min_value=1, max_value=50, value=20, step=1)
    bb_flag = st.sidebar.checkbox(label="볼린저 밴드 추가")
    bb_periods = st.sidebar.number_input("볼린저 밴드 기간", min_value=1, max_value=50, value=20, step=1)
    bb_std = st.sidebar.number_input("표준편차 수", min_value=1, max_value=4, value=2, step=1)
    rsi_flag = st.sidebar.checkbox(label="RSI 추가")
    rsi_periods = st.sidebar.number_input("RSI 기간", min_value=1, max_value=50, value=14, step=1)

    # 데이터 로드
    df = load_data(ticker, start_date, end_date)

    # 데이터 미리보기
    data_exp = st.expander("데이터 미리보기")
    available_cols = df.columns.tolist()
    columns_to_show = data_exp.multiselect("열", available_cols, default=available_cols)
    data_exp.dataframe(df[columns_to_show])

    csv_file = convert_df_to_csv(df[columns_to_show])
    data_exp.download_button(
        label="선택한 데이터를 CSV로 다운로드",
        data=csv_file,
        file_name=f"{ticker}_stock_prices.csv",
        mime="text/csv",
    )

    # 차트 생성
    st.plotly_chart(create_stock_chart(df, volume_flag, sma_flag, sma_periods, bb_flag, bb_periods, bb_std, rsi_flag, rsi_periods), use_container_width=True)

    # ARIMA 분석
    st.header("ARIMA 모델을 이용한 주가 예측")
    if st.button("ARIMA 분석 수행"):
        with st.spinner("ARIMA 분석 중..."):
            forecast_30, forecast_7, summary, arima_trend, percent_change = perform_arima_analysis(df)
            if forecast_30 is not None and forecast_7 is not None:
                if arima_trend == "상승":
                    st.success(f"ARIMA 분석 결과, 향후 30일 동안 상승 추세가 예상됩니다. (예상 변화: {percent