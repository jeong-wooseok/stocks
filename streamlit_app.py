import yfinance as yf
import streamlit as st
import datetime 
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pmdarima import auto_arima
import numpy as np

# (기존의 함수들은 그대로 유지)

# 기술적 지표 계산 함수
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

# 메인 부분
st.title("티커 기술적 분석 웹 서비스")

# (사이드바 및 데이터 로드 부분은 그대로 유지)

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
st.header("Auto ARIMA 분석")
if st.button("Auto ARIMA 분석 수행"):
    with st.spinner("Auto ARIMA 분석 중..."):
        model = auto_arima(df['Close'], start_p=1, start_q=1, max_p=3, max_q=3, m=1,
                           start_P=0, seasonal=False, d=1, D=1, trace=True,
                           error_action='ignore', suppress_warnings=True, stepwise=True)
        
        # 모델 요약
        st.subheader("ARIMA 모델 요약")
        st.text(str(model.summary()))
        
        # 예측 및 시각화
        n_periods = 30  # 30일 예측
        fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
        index_of_fc = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=n_periods, freq='D')

        # 결과 시각화
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='실제 가격'))
        fig.add_trace(go.Scatter(x=index_of_fc, y=fc, mode='lines', name='예측', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=index_of_fc.tolist()+index_of_fc.tolist()[::-1], 
                                 y=confint[:,1].tolist()+confint[:,0].tolist()[::-1],
                                 fill='toself',
                                 fillcolor='rgba(255,0,0,0.1)',
                                 line=dict(color='rgba(255,0,0,0.1)'),
                                 name='신뢰 구간'))
        
        fig.update_layout(title='주가와 ARIMA 예측',
                          xaxis_title='날짜',
                          yaxis_title='가격',
                          height=600)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 트렌드 존재 여부 판단
        trend_diff = np.diff(fc)
        trend_direction = np.mean(trend_diff)
        
        if abs(trend_direction) > 0.01:  # 임계값 설정 (필요에 따라 조정 가능)
            trend_message = "상승" if trend_direction > 0 else "하락"
            st.success(f"분석 결과, 향후 30일 동안 {trend_message} 트렌드가 예상됩니다.")
        else:
            st.info("분석 결과, 향후 30일 동안 뚜렷한 트렌드가 없을 것으로 예상됩니다.")

# (기존의 코드는 그대로 유지)
