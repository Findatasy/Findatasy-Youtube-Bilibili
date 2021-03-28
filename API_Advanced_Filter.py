# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 2021

@author: findatasy
"""


from futu import *
import pandas as pd
from datetime import datetime, timedelta
import time

#%%

# =============================================================================
# Technique 1: multiple filters (for all stocks)
# =============================================================================
quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)

# Find stocks with increasing turnover in a 20/3 days window

# create 1st filter: 20 days average turnover between $30-50m
first_filter = AccumulateFilter()
first_filter.filter_min = 30000000
first_filter.filter_max = 50000000
first_filter.stock_field = StockField.TURNOVER
first_filter.days = 20
first_filter.is_no_filter = False

# create 2nd filter: 3 days average turnover between $70-90m
second_filter = AccumulateFilter()
second_filter.filter_min = 70000000
second_filter.filter_max = 90000000
second_filter.stock_field = StockField.TURNOVER
second_filter.days = 3
second_filter.is_no_filter = False

# apply the 2 filters
ret, ls = quote_ctx.get_stock_filter(market=Market.HK, filter_list=[first_filter, second_filter])
if ret == RET_OK:
    last_page, all_count, ret_list = ls
    for item in ret_list:
        print(item.stock_code, item.stock_name)
else:
    print('error: ', ls)

quote_ctx.close()

#%%

# =============================================================================
# Technique 2: sector filter
# =============================================================================
quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)

# Find sectors increased more than 5% in the last 3 days

# today's date
today = datetime.today()
# to overcome holidays, use a 20 calendar days window to cover 3 trading days
ten_days_ago = today - timedelta(19)
# find the trading days
ret, date = quote_ctx.request_trading_days(TradeDateMarket.HK, 
                                           start=ten_days_ago.strftime('%Y-%m-%d'), 
                                           end=today.strftime('%Y-%m-%d'))
# get all sector list
ret, data = quote_ctx.get_plate_list(Market.HK, Plate.ALL)

# return kline of all sectors in the last 3 days
for code, plate_name in zip(data.code, data.plate_name):
    # date use last 3 trading days
    ret, stock_data, page_req_key = quote_ctx.request_history_kline(code, 
                                                                    start=date[-3]['time'], 
                                                                    end=date[-1]['time'])
    time.sleep(0.5)
    # filter the sectors with a 5% increment
    if (stock_data.change_rate.cumsum() >= 5).any():
        print(code, plate_name)  

quote_ctx.close()

#%%
# =============================================================================
# Technique 3: Technique 1 + candlestick patterns
# =============================================================================
quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)

# get all candlestick patterns from talib module
import talib
candle_names = talib.get_function_groups()['Pattern Recognition']

# save the filtered stocks from Part 1 into stock_list
stock_list = []
for item in ret_list:
    stock_list.append(item.stock_code)

# set the date range of last 30 days
today = datetime.today()
start_date = today - timedelta(29)

# create a blank dictionary to save data later
Stock_Info = {}
# get history kline of our stock list
for i in stock_list:
    ret, data, page_req_key = quote_ctx.request_history_kline(i, 
                                                              start=start_date.strftime('%Y-%m-%d'), 
                                                              end=today.strftime('%Y-%m-%d'))
    # extract OHLC 
    op = data['open']
    hi = data['high']
    lo = data['low']
    cl = data['close']
    # time.sleep(1)
    # generate candlestick patterns
    for candle in candle_names:
        data[candle] = getattr(talib, candle)(op, hi, lo, cl)
    # save into Stock_Info
    Stock_Info[i] = pd.DataFrame(data)


# further filter the stocks have a candlestick pattern
for key, item in Stock_Info.items():
    for i in candle_names:
        # print if last day has candlestick pattern signal
        if (item[i].iloc[-1] != 0):
            print(key, item[i].name, item[i].iloc[-1])

quote_ctx.close()