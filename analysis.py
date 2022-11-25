import streamlit as st
import yfinance as yf
from plotly import graph_objs as go
from prophet import Prophet
from prophet.plot import plot_plotly
import numpy as np

from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def home():
    st.header('Begin exploring the data using the menu on the left')

@st.cache
def load_data(START, TODAY, ticker):
    # will return data in pandas data frame
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


def plot_raw_data(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],
                  y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'],
                  y=data['Close'], name='stock_close'))
    fig.layout.update(title_text="Time Series Data",
                      xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


def printRawData(data):
    st.subheader('Raw data')
    st.write(data)


# Forecasting with prophet
def futureMovementData(data, period):
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()  # changepoint_range=1,weekly_seasonality=True, changepoint_prior_scale=0.10)
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)
    return forecast


# Forecasting with prophet(Visualisation)
def futureMovementVisualisation(data, period):
    
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()  # changepoint_range=1,weekly_seasonality=True, changepoint_prior_scale=0.10)
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    st.write('Forecast Graph')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    st.write('Forecast components')
    fig2 = m.plot_components(forecast)
    st.write(fig2)


# ANALYSIS PURPOSE
def analysis(forecast, ticker, START):
    TODAY = "2018-04-01"
    data = yf.download(ticker, START, TODAY) #will return data in pandas data frame
    data.reset_index(inplace=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'], name='original'))
    fig.add_trace(go.Scatter(x=forecast['ds'],y=(forecast['yhat_upper']+forecast['yhat_lower'])/2, name='predicted'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


# Determining Accuracy
def accuracy(START, TODAY, forecast, ticker):
    START = "2017-01-01"
    TODAY = "2022-01-01"
    # will return data in pandas data frame
    testdata = yf.download(ticker, START, TODAY)
    testdata.reset_index(inplace=True)
    df_test = testdata[['Date', 'Close']]
    df_test = df_test.rename(columns={"Date": "ds", "Close": "y"})

    metric_df = forecast.set_index('ds')[['yhat']].join(
        df_test.set_index('ds').y).reset_index()

    metric_df = metric_df[metric_df['y'].notna()]

    metric_df = metric_df.tail(30)

    #st.write(mean_absolute_percentage_error(metric_df.y, metric_df.yhat))

    # st.write(forecast.tail())
    # st.write(df_test.tail())
    # st.write(metric_df)
    # st.write(mean_squared_error(metric_df.y, metric_df.yhat))
    # st.write(r2_score(metric_df.y, metric_df.yhat))
    # st.write(mean_absolute_error(metric_df.y, metric_df.yhat))


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# TREND
def fetchNews(ticker):
    finviz_url = 'https://finviz.com/quote.ashx?t='

    url = finviz_url + ticker

    req = Request(url=url, headers={'user-agent': 'stock-p'})
    response = urlopen(req)

    html = BeautifulSoup(response, 'html')
    news_table = html.find(id='news-table')


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


    s_df = pd.DataFrame(parsed_data, columns=['date', 'time', 'title'])
    return s_df


def sentimentAnalysis(s_df):
    vader = SentimentIntensityAnalyzer()

    def f(title): return vader.polarity_scores(title)['compound']

    s_df['compound'] = s_df['title'].apply(f)
    #s_df['date'] = pd.to_datetime(s_df.date).dt.date

    # plt.figure(figsize=(10,8))

    mean_df = s_df.groupby(['date']).mean()
    # mean_df = mean_df.unstack()
    # mean_df = mean_df.xs('compound', axis="columns").transpose()
    # mean_df.plot(kind='bar')

    return mean_df


def recommendedAction(data, s_df, period):
    forecast = futureMovementData(data, period)
    originalDataLatestVal = data['Close'].iloc[-1]
    forecastDataLatestVal = forecast['yhat'].iloc[-1]
    st.write("\nA N A L Y S I S")
    st.write("Buying Price : ",originalDataLatestVal)
    st.write("Selling Price considered : ",forecastDataLatestVal)

    mean_df = sentimentAnalysis(s_df)
    averageSentiment = mean_df["compound"].mean()

    years = period/365

    profitLoss = forecastDataLatestVal-originalDataLatestVal
    st.write("Profit/Loss expected : ",profitLoss,"$")
    returnPercent = (profitLoss*100/originalDataLatestVal)/years

    if( returnPercent > 11 ):
        st.write(
            "Future Movement of the stock looks promising. Returns expected around ",
            returnPercent,
            "%."
        )

        if(averageSentiment > 0):
            st.write("Good Time to invest in this stock.")
            action = '<p style="font-family:algerian; color:Green; font-size: 32px;">Recommended Action: BUY</p>'
            st.markdown(action, unsafe_allow_html=True)
        else:
            st.write("But stock might go further down due to the negativity surrounding around the stock. So wait for the dip")
            action = '<p style="font-family:algerian; color:yellow; font-size: 32px;">Recommended Action: WAIT FOR THE RIGHT MOMENT AND BUY</p>'
            st.markdown(action, unsafe_allow_html=True)

    else:
        st.write(
            "Returns expected around ",
            returnPercent,
            "%."
        )
        action = '<p style="font-family:algerian; color:red; font-size: 32px;">Recommended Action: AVOID</p>'
        st.markdown(action, unsafe_allow_html=True)
    
    disclaimer = '<marquee style="font-size: 11px;" scrollamount="15d">Investment in stock market is subject to market risks, read all the related documents carefully before investing.</marquee>'
    st.markdown(disclaimer, unsafe_allow_html=True)


# def allInOneAnalysis(data, period, s_df):
#     printRawData(data)
#     plot_raw_data(data)
#     st.subheader('Forecast data')
#     forecast = futureMovementData(data, period)
#     st.write(forecast)
#     futureMovementVisualisation(data, period)
#     st.write("News from famous publications")
#     st.write(s_df)
#     mean_df = sentimentAnalysis(s_df)
#     st.bar_chart(mean_df, use_container_width=True)
#     st.write(mean_df)
#     recommendedAction(data, s_df, period)