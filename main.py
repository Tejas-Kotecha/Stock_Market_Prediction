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

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

from analysis import *;

st.set_page_config(layout="wide")


START = "2012-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")
st.text('This is a web app which allows you to predict stock movements and to give analysis of sentiment of stocks')


# Sidebar setup
st.sidebar.title('Sidebar')
tickers = pd.read_html(
    'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
stockdownload = pd.read_html(
    'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
stocks_list = tickers.Symbol.to_list()
company_list = tickers.Security.to_list()

res = [i + " -- " + j for i, j in zip(stocks_list, company_list)]
spl = st.sidebar.selectbox("Select dataset for prediction", res).split(" -- ")
selected_stock = spl[0]

n_years = st.sidebar.slider("Years of prediction:", 1, 5)
period = n_years * 365

data_load_state = st.text("Load data...")
data = load_data(START, TODAY, selected_stock)
data_load_state.text("Data Load Success")
s_df = fetchNews(selected_stock)

# if st.sidebar.button('Load Data'):
#   data_load_state = st.text("Load data...")
#   data = load_data(START, TODAY, selected_stock)
#   data_load_state.text("Data Load Success")
#   s_df = fetchNews(selected_stock)

#Sidebar navigation
st.sidebar.title('Navigation')
options = st.sidebar.radio('Select what you want to display:', [
  'Home',
  'Scout Raw Data',
  'Visualise Raw Data',
  'Scout Future Movement Data',
  'Visualise Future Movement Data',
  'Fetch News of stock',
  'Analyze sentiment of stock',
  'Recommended Action for the stock',
])

# Navigation options
if options == 'Home':
  home()
elif options == 'Scout Raw Data':
  printRawData(data)
elif options == 'Visualise Raw Data':
  plot_raw_data(data)
elif options == 'Scout Future Movement Data':
  st.subheader('Forecast data')
  forecast = futureMovementData(data, period)
  st.write(forecast)
elif options == 'Visualise Future Movement Data':
  futureMovementVisualisation(data, period)
elif options == 'Fetch News of stock':
  st.write("News from famous publications")
  st.write(s_df)
elif options == 'Analyze sentiment of stock':
  mean_df = sentimentAnalysis(s_df)
  st.bar_chart(mean_df, use_container_width=True)
  st.write(mean_df)
elif options == 'Recommended Action for the stock':
  recommendedAction(data, s_df, period)
# elif options == 'All in one Analysis':
#   allInOneAnalysis(data, period, s_df)