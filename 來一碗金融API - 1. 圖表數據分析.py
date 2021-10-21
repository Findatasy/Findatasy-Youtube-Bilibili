# -*- coding: utf-8 -*-

#pip install plotly
import plotly.graph_objects as go
from plotly.offline import plot
import pandas as pd
#每次Apple產品公佈大會的日期
iphone_dict = {'Date':['2007-01-09','2008-06-09','2009-06-08','2010-06-07','2011-10-04',
                       '2012-09-12','2013-09-10','2014-09-09','2015-09-09','2016-03-21',
                       '2016-09-07','2017-09-12','2018-09-12','2019-09-10','2020-04-17',
                       '2020-10-13','2021-09-14'],
               'Model':['iPhone','iPhone 3G','iPhone 3GS','iPhone 4','iPhone 4S','iPhone 5',
                        'iPhone 5S','iPhone 6','iPhone 6S','SE','iPhone 7','iPhone X/8',
                        'iPhone XS/XR','iPhone 11','iPhone SE2','iPhone 12','iPhone 13']}
#轉變為DataFrame
iphone = pd.DataFrame(iphone_dict)
#轉日期為Index
iphone = iphone.set_index(['Date'])

#2006-09-01 to 2021-09-15
df1 = pd.read_csv('https://query1.finance.yahoo.com/v7/finance/download/AAPL?period1=1157068800&period2=1631750400&interval=1d&events=history&includeAdjustedClose=true')



#2006-09-01 to 2021-09-15, all 16 iphones
df = df1
df = df.set_index(['Date'])
#df = df.loc['2006-9-1':'2013-1-1']
fig = go.Figure(data=[go.Candlestick(x=df.index,
                                     open=df['Open'],
                                     high=df['High'],
                                     low=df['Low'],
                                     close=df['Close'])])
fig.update_layout(
    title = 'AAPL Price', title_x=0.5,
    yaxis_title = 'Stock Price ($)',
)

for i in range(0,17):
  fig.add_annotation(x=iphone.index[i], y=df['High'].loc[iphone.index[i]],
            text=iphone['Model'].iloc[i],
            showarrow=True,
            arrowhead=1)

plot(fig, auto_open=True)



#2006-09-01 to 2012-01-01 Steve Jobs Era
df = df1
df = df.set_index(['Date'])
df = df.loc['2006-09-01':'2012-01-01']

fig = go.Figure(data=[go.Candlestick(x=df.index,
                                     open=df['Open'],
                                     high=df['High'],
                                     low=df['Low'],
                                     close=df['Close'])])
fig.update_layout(
    title = 'The first 5 iPhones (Steve Jobs Era)', title_x=0.5,
    yaxis_title = 'Stock Price ($)',
)

for i in range(0,5):
  fig.add_annotation(x=iphone.index[i], y=df['High'].loc[iphone.index[i]],
            text=iphone['Model'].iloc[i],
            showarrow=True,
            arrowhead=1)

plot(fig, auto_open=True)

#2012-09-01 to 2016-07-01 Tim Cook Era (first)
df = df1
df = df.set_index(['Date'])
df = df.loc['2012-09-01':'2016-07-01']

fig = go.Figure(data=[go.Candlestick(x=df.index,
                                     open=df['Open'],
                                     high=df['High'],
                                     low=df['Low'],
                                     close=df['Close'])])
fig.update_layout(
    title = 'The next 5 iPhones (rise of Tim Cook Era)', title_x=0.5,
    yaxis_title = 'Stock Price ($)',
)

for i in range(5,10):
  fig.add_annotation(x=iphone.index[i], y=df['High'].loc[iphone.index[i]],
            text=iphone['Model'].iloc[i],
            showarrow=True,
            arrowhead=1)

plot(fig, auto_open=True)

#2016-09-01 to now Tim Cook Era (2nd 5 years)
df = df1
df = df.set_index(['Date'])
df = df.loc['2016-09-01':]

fig = go.Figure(data=[go.Candlestick(x=df.index,
                                     open=df['Open'],
                                     high=df['High'],
                                     low=df['Low'],
                                     close=df['Close'])])
fig.update_layout(
    title = 'The new Tim Cook Era', title_x=0.5,
    yaxis_title = 'Stock Price ($)',
)

for i in range(10,17):
  fig.add_annotation(x=iphone.index[i], y=df['High'].loc[iphone.index[i]],
            text=iphone['Model'].iloc[i],
            showarrow=True,
            arrowhead=1)

plot(fig, auto_open=True)

#calculate performance
df = df1
df = df.set_index(['Date'])
col1 = [] #Date
col2 = [] #Model
col3 = [] #Percentage increase in 20/40/60 days

for i in iphone.index:
    col1.append(i)
    col2.append(iphone['Model'].loc[i])

for i in range(0,17):
    col3.append((df.shift(periods=-19, axis=0).loc[iphone.index[i]]['Close'] / df.loc[iphone.index[i]]['Close'])-1) 

col_tuples = list(zip(col1,col2,col3))    
perf = pd.DataFrame(col_tuples,columns=['Date','Model','Performance'])