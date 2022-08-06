from dataclasses import dataclass
import math
import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2005-01-01"
TODAY = "2016-01-01"
#TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")

stocks = ("SPY","AAPL","GOOG","MSFT","GME","TSLA","HDFCBANK.NS")
selected_stock = st.selectbox("Select dataset for prediction",stocks)

n_years = st.slider("Months of prediction:", 2, 5)
period = n_years * 365

@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY) #will return data in pandas data frame
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load data...")
data = load_data(selected_stock)
data_load_state.text("Loading data...done!")

# st.subheader('Raw data')
# st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'], name='stock_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

#Forecasting with prophet
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close":"y"})

m = Prophet()#changepoint_range=1,weekly_seasonality=True, changepoint_prior_scale=0.10)
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Forecast data')
st.write(forecast.tail())

st.write('forecast data')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

TODAY = "2018-04-01"
data = yf.download(selected_stock, START, TODAY) #will return data in pandas data frame
data.reset_index(inplace=True)
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'], name='original'))
fig.add_trace(go.Scatter(x=forecast['ds'],y=(forecast['yhat_upper']+forecast['yhat_lower'])/2, name='predicted'))
fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
st.plotly_chart(fig)

# st.write('forecast components')
# fig2 = m.plot_components(forecast)
# st.write(fig2)

#Testing
st.write(forecast.tail())