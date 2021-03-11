# -*- coding: utf-8 -*-
"""
"""


import talib
import pandas as pd
import numpy as np
from futu import *

#先拿取所有可以自動辨識的Candlestick形態
candle_names = talib.get_function_groups()['Pattern Recognition']

#參考https://www.investopedia.com/articles/active-trading/092315/5-most-powerful-candlestick-patterns.asp
Best_five_candlestick_pattern = ['CDL3LINESTRIKE', 'CDLXSIDEGAP3METHODS', 'CDL3BLACKCROWS', 'CDLEVENINGSTAR', 'CDLABANDONEDBABY']

#連接富途伺服器
quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)

#自取stock_list - 可以自選你喜歡的股票
stock_list = ['HK.00941', 'HK.00700', 'HK.09988']

#先開一個空的Dictionary去放資料
Stock_Info = {}   

# Loop我們的自取stock_list
for i in stock_list:
    #拿取股票資料
    ret, data, page_req_key = quote_ctx.request_history_kline(i, start='2021-01-02', end='2021-02-28')
    # extract OHLC 
    op = data['open']
    hi = data['high']
    lo = data['low']
    cl = data['close']
    # Loop我們的Candlestick形態﹐這裡我們先用所有可用的形態
    for candle in candle_names:
        #在TALIB的Module拿取candle的資料
        data[candle] = getattr(talib, candle)(op, hi, lo, cl)
    #把我們拿取的數據放進上面空的Dictionary
    Stock_Info[i] = pd.DataFrame(data)

# 在Dictionary拿取資料
for key, item in Stock_Info.items():
    #Loop我們的Candlestick名單
    for i in candle_names:
        #如果行數內有不等於0的數據
        if (item[i].iloc[-1] != 0):
            #把相關的stock code跟他的名字拿出來
            print(key, item[i].name)
            
     