from dataclasses import dataclass
import math
import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt

START = "2015-01-01"
#TODAY = "2016-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")

stocks = ("AMD","AMZN","SPY","GOOG","MSFT","GME","TSLA","HDFCBANK.NS")
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

m = Prophet() #changepoint_range=1,weekly_seasonality=True, changepoint_prior_scale=0.10)
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Forecast data')
st.write(forecast.tail())

st.write('forecast data')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

#-------------------------------------ANALYSIS PURPOSE----------------------------------------------------
# TODAY = "2018-04-01"
# data = yf.download(selected_stock, START, TODAY) #will return data in pandas data frame
# data.reset_index(inplace=True)
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'], name='original'))
# fig.add_trace(go.Scatter(x=forecast['ds'],y=(forecast['yhat_upper']+forecast['yhat_lower'])/2, name='predicted'))
# fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
# st.plotly_chart(fig)

# st.write('forecast components')
# fig2 = m.plot_components(forecast)
# st.write(fig2)

#--------------------------------Testing------------------------------
#st.write(forecast.tail())


#----------------------------------------------TREND-------------------------------------------
finviz_url = 'https://finviz.com/quote.ashx?t='

url = finviz_url + selected_stock

req = Request(url=url, headers={'user-agent': 'stock-p'})
response = urlopen(req)

html = BeautifulSoup(response,'html')
news_table = html.find(id='news-table')

#print(news_table)

rows = news_table.findAll('tr')

parsed_data = []

for index, row in enumerate(rows):
  title = row.a.get_text()
  date_data = row.td.text.split(' ')

  if len(date_data) == 1:
    time = date_data[0]
  else:
    date = date_data[0]
    time = date_data[1]

  parsed_data.append([date, time, title])
  
s_df = pd.DataFrame(parsed_data,columns=['date','time','title'])

vader = SentimentIntensityAnalyzer()

f = lambda title: vader.polarity_scores(title)['compound']
s_df['compound'] = s_df['title'].apply(f)
#s_df['date'] = pd.to_datetime(s_df.date).dt.date

#plt.figure(figsize=(10,8))

mean_df = s_df.groupby(['date']).mean()
# mean_df = mean_df.unstack()
# mean_df = mean_df.xs('compound', axis="columns").transpose()
# mean_df.plot(kind='bar')
st.bar_chart(mean_df, use_container_width=True)
st.write(mean_df)
