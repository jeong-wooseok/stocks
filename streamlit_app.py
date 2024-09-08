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
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from itertools import product

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

@st.cache_data(ttl=3600, show_spinner=False)
def load_data(symbol, start, end, retries=3):
    for attempt in range(retries):
        try:
            data = yf.download(symbol, start=start, end=end, progress=False)
            if data.empty:
                raise ValueError(f"No data available for {symbol}")
            return data
        except Exception as e:
            if attempt < retries - 1:
                st.warning(f"데이터 로딩 중 오류 발생. 재시도 중... (시도 {attempt + 1}/{retries})")
                sleep(2)  # 재시도 전 잠시 대기
            else:
                st.error(f"데이터를 불러오는 데 실패했습니다. 오류: {str(e)}")
                return pd.DataFrame()  # 빈 DataFrame 반환

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

def calculate_volatility(returns, window=20):
    return returns.rolling(window=window).std() * np.sqrt(252)
    
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import streamlit as st
from itertools import product

def perform_arima_analysis(data):
    try:
        if 'Close' not in data.columns:
            return None, None, "'Close' 열이 데이터에 없습니다.", None, None, None
        
        close_data = data['Close'].dropna()
        
        if len(close_data) < 30:
            return None, None, "데이터가 충분하지 않습니다. 최소 30일 이상의 데이터가 필요합니다.", None, None, None
        
        # 로그 수익률 계산
        log_returns = np.log(close_data / close_data.shift(1)).dropna().values.flatten()  # 1차원 배열로 변환
        
        # 그리드 서치를 통한 최적 파라미터 찾기
        best_aic = np.inf
        best_order = None
        for p, d, q in product(range(0, 3), [1], range(0, 3)):
            try:
                model = ARIMA(log_returns, order=(p, d, q))
                results = model.fit()
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = (p, d, q)
            except Exception as e:
                st.write(f"파라미터 (p,d,q)=({p},{d},{q}) 시도 중 오류 발생: {str(e)}")
                continue
        
        if best_order is None:
            raise ValueError("적합한 ARIMA 모델을 찾을 수 없습니다.")
        
        # 최적의 파라미터로 ARIMA 모델 적합
        model = ARIMA(log_returns, order=best_order)
        results = model.fit()
        
        summary = str(results.summary())
        
        # 로그 수익률 예측
        forecast_returns = results.forecast(steps=30)
        
        # 예측된 로그 수익률을 가격으로 변환
        last_price = close_data.iloc[-1]
        forecast_prices = last_price * np.exp(np.cumsum(forecast_returns))
        forecast_30 = pd.Series(forecast_prices, index=pd.date_range(start=close_data.index[-1] + pd.Timedelta(days=1), periods=30))
        forecast_7 = forecast_30[:7]
        
        # 변동성 계산
        volatility = pd.Series(log_returns).rolling(window=20).std() * np.sqrt(252)
        current_volatility = volatility.iloc[-1]
        
        # 추세 판단
        percent_change = ((forecast_30.iloc[-1] - last_price) / last_price) * 100
        
        if percent_change > 5:
            trend = "상승"
        elif percent_change < -5:
            trend = "하락"
        else:
            trend = "횡보"
        
        # 예측 결과 유효성 검사
        if np.isnan(forecast_30).any() or np.isnan(forecast_7).any():
            raise ValueError("예측 결과에 NaN 값이 포함되어 있습니다.")
        
        if abs(percent_change) < 0.01:
            raise ValueError("예측 변화율이 비정상적으로 작습니다. 모델을 재검토해야 합니다.")
        
        return forecast_30, forecast_7, summary, trend, percent_change, current_volatility
    
    except Exception as e:
        error_msg = f"ARIMA 분석 중 오류가 발생했습니다: {str(e)}"
        st.error(error_msg)
        return None, None, error_msg, None, None, None


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


def create_stock_chart(df, volume_flag, sma_flag, sma_periods, bb_flag, bb_periods, bb_std, rsi_flag, rsi_periods, ticker, tickers_companies_dict):
    # 서브플롯 개수 결정 (주가는 항상 포함)
    subplot_count = 1 + volume_flag + rsi_flag
    row_heights = [0.5] + [0.25] * (subplot_count - 1)
    
    fig = make_subplots(rows=subplot_count, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=row_heights)

    # 메인 캔들스틱 차트
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

    current_row = 2

    # 거래량 차트
    if volume_flag:
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', 
                             marker_color=color_palette['volume']), row=current_row, col=1)
        fig.update_yaxes(title_text="Volume", row=current_row, col=1)
        current_row += 1

    # RSI 차트
    if rsi_flag:
        df = add_rsi(df, rsi_periods)
        fig.add_trace(go.Scatter(x=df.index, y=df[f'RSI_{rsi_periods}'], name=f'RSI {rsi_periods}', 
                                 line=dict(color=color_palette['rsi'], width=1)), row=current_row, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1)
        fig.update_yaxes(title_text="RSI", row=current_row, col=1)

    fig.update_layout(
        title=f"{tickers_companies_dict[ticker]}'s stock price",
        height=300 * subplot_count,  # 각 서브플롯의 높이를 300px로 설정
        plot_bgcolor=color_palette['background'],
        paper_bgcolor=color_palette['background'],
        font_color=color_palette['text'],
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=color_palette['grid'])
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=color_palette['grid'])

    # 각 서브플롯에 대해 y축 범위 및 제목 설정
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(range=[df['Low'].min()*0.95, df['High'].max()*1.05], row=1, col=1)
    
    if volume_flag:
        fig.update_yaxes(range=[0, df['Volume'].max()*1.1], row=2, col=1)
    
    if rsi_flag:
        fig.update_yaxes(range=[0, 100], row=subplot_count, col=1)

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
    # 사용자 매뉴얼
    st.write("""
    ## 사용자 매뉴얼
    1. 사이드바에서 분석하고 싶은 S&P 500 주식을 선택하세요.
    2. 분석하고 싶은 기간의 시작일과 종료일을 선택하세요.
    3. 원하는 기술적 지표(거래량, SMA, 볼린저 밴드, RSI)를 선택하고 매개변수를 조정하세요.
    4. 차트를 통해 주가의 움직임과 기술적 지표를 확인하세요.
    5. 'ARIMA 분석 수행' 버튼을 클릭하여 향후 30일간의 가격 예측과 추세 분석 결과를 확인하세요.
    6. '시계열 분해 수행' 버튼을 클릭하여 상세한 시계열 분석 결과를 확인하세요.
    7. 계절성, 잔차, 정상성 검정 결과를 통해 주가의 특성을 파악하세요.
    """)
    # 사이드바 설정
    st.sidebar.header("주식 매개변수")
    available_tickers, tickers_companies_dict = get_sp500_components()
    ticker = st.sidebar.selectbox("티커", available_tickers, format_func=tickers_companies_dict.get)
    
    # 시작일을 1년 전으로 설정
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=365)
    
    start_date = st.sidebar.date_input("시작일", start_date)
    end_date = st.sidebar.date_input("종료일", end_date)

    if start_date > end_date:
        st.sidebar.error("종료일은 시작일 이후여야 합니다.")
        return

    # 데이터 로드
    with st.spinner("데이터 로딩 중..."):
        df = load_data(ticker, start_date, end_date)

    if df.empty:
        st.warning("선택한 티커에 대한 데이터를 불러올 수 없습니다. 다른 티커를 선택해 주세요.")
        return
        
    st.sidebar.header("기술적 분석 매개변수")
    volume_flag = st.sidebar.checkbox(label="거래량 추가", value=True)
    sma_flag = st.sidebar.checkbox(label="SMA 추가", value=True)
    sma_periods = st.sidebar.number_input("SMA 기간", min_value=1, max_value=50, value=20, step=1)
    bb_flag = st.sidebar.checkbox(label="볼린저 밴드 추가", value=True)
    bb_periods = st.sidebar.number_input("볼린저 밴드 기간", min_value=1, max_value=50, value=20, step=1)
    bb_std = st.sidebar.number_input("표준편차 수", min_value=1, max_value=4, value=2, step=1)
    rsi_flag = st.sidebar.checkbox(label="RSI 추가", value=True)
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
    st.plotly_chart(create_stock_chart(df, volume_flag, sma_flag, sma_periods, bb_flag, bb_periods, bb_std, rsi_flag, rsi_periods, ticker, tickers_companies_dict), use_container_width=True)

    # main 함수 내 ARIMA 분석 부분
    st.header("ARIMA 모델을 이용한 주가 예측")
    if st.button("ARIMA 분석 수행"):
        with st.spinner("ARIMA 분석 중..."):
            forecast_30, forecast_7, summary, arima_trend, percent_change, current_volatility = perform_arima_analysis(df)
            if forecast_30 is not None and forecast_7 is not None:
                st.subheader(f"ARIMA 분석 결과: {arima_trend}")
                st.info(f"향후 30일 동안의 예상 변화율: {percent_change:.2f}%")
                st.info(f"현재 연간 변동성: {current_volatility*100:.2f}%")
                
                # ARIMA 예측 결과 시각화
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='실제 가격', line=dict(color=color_palette['log_data'])))
                fig.add_trace(go.Scatter(x=forecast_30.index, y=forecast_30, mode='lines', name='30일 예측', line=dict(color=color_palette['forecast'])))
                fig.add_trace(go.Scatter(x=forecast_7.index, y=forecast_7, mode='lines', name='7일 예측', line=dict(color='green')))
                
                fig.update_layout(title='ARIMA 모델 예측 결과',
                                  xaxis_title='날짜',
                                  yaxis_title='가격',
                                  height=500,
                                  plot_bgcolor=color_palette['background'],
                                  paper_bgcolor=color_palette['background'],
                                  font_color=color_palette['text'],
                                  hovermode='x unified')
                
                st.plotly_chart(fig, use_container_width=True)

                # 7일 예측 결과를 표 형태로 표시
                st.subheader("향후 7일간 예측 가격")
                forecast_df = pd.DataFrame({
                    '날짜': forecast_7.index,
                    '예측 가격': forecast_7.values.round(2)
                })
                st.table(forecast_df)
            else:
                st.warning(f"ARIMA 분석을 수행할 수 없습니다. 이유: {summary}")
    
    

    # 시계열 분해 섹션
    st.header("시계열 분해 분석")
    if st.button("시계열 분해 수행"):
        with st.spinner("시계열 분해 중..."):
            log_data, diff_data, trend, seasonal, residual = perform_time_series_decomposition(df['Close'])
            
            st.plotly_chart(create_decomposition_chart(log_data, diff_data, trend, seasonal, residual), use_container_width=True)
            
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

    # 추가 정보
    st.write("""
    ## 추가 정보
    - 이 앱은 S&P 500 구성 주식에 대한 기술적 분석을 제공합니다.
    - 사용된 데이터는 Yahoo Finance에서 실시간으로 가져오며, 최신 정보를 반영합니다.
    - 시계열 분해는 주가의 추세, 계절성, 잔차 요소를 분리하여 분석합니다.
    - ARIMA 모델은 과거 데이터를 바탕으로 미래 가격을 예측합니다. 단, 이는 참고용이며 실제 투자 결정에는 다양한 요소를 고려해야 합니다.
    - ADF 테스트는 시계열 데이터의 정상성을 검정합니다. p-value가 0.05 미만이면 정상성을 가정할 수 있습니다.
    """)

    # 면책조항
    st.write("""
    ## 면책조항
    이 앱에서 제공하는 정보는 교육 및 참고 목적으로만 사용되어야 합니다. 실제 투자 결정은 본인의 판단하에 이루어져야 하며, 
    이 앱의 분석 결과로 인한 투자 손실에 대해 개발자는 책임을 지지 않습니다.
    """)

if __name__ == "__main__":
    main()