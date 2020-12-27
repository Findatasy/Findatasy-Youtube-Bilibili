# -*- coding: utf-8 -*-
"""
@author: findatasy
"""

from futu import *
import pandas as pd
import seaborn as sns

# pip install yfinance
import yfinance as yf
import matplotlib
import matplotlib.pyplot as plt
#防止中文亂碼
matplotlib.rcParams['font.sans-serif'] = ['SimHei']

# %%
# =============================================================================
# 1. 金價和美股，港股分析
# =============================================================================

# =============================================================================
# 通過yahoo finance 拿歷史數據
# =============================================================================

# 怎麼找ticker？ google 'gold price yahoo finance'

# 拿紐約金的歷史價格
# 先拿取國際黃金代碼
GOLD = yf.Ticker('GC=F')
# 然後再拿取時間區間的國際黃金價格
GOLD = GOLD.history(start='2000-01-01', end='2020-12-12')
# 只留下收盤價
GOLD = GOLD.Close
# column 名字由Close改成Gold
GOLD.rename("GOLD", inplace=True)

# 拿道瓊斯的價格
DJI = yf.Ticker('^DJI')
DJI = DJI.history(start='2000-01-01', end='2020-12-12')
DJI = DJI.Close
DJI.rename("DJI", inplace=True)

# 拿恆指的價格
HSI = yf.Ticker('^HSI')
HSI = HSI.history(start='2000-01-01', end='2020-12-12')
HSI = HSI.Close
HSI.rename("HSI", inplace=True)


# %%
# =============================================================================
# 查看數據完整性
# =============================================================================

# 查看頭幾行
GOLD.head()
DJI.head()
HSI.head()
# 查看最末幾行
GOLD.tail()
DJI.tail()
HSI.tail()

# %%
# =============================================================================
# 數據清理和畫圖
# =============================================================================

GOLD_INDEX = pd.merge(GOLD, DJI, left_index=True, right_index=True)
GOLD_INDEX = GOLD_INDEX.merge(HSI, left_index=True, right_index=True)
# 把沒有數據的日期拿走
GOLD_INDEX_100 = GOLD_INDEX.dropna()
# 利用lamdba﹐把每一行的數據以x/x[0]*100公式運算一次
GOLD_INDEX_100 = GOLD_INDEX_100.apply(lambda x: x/x[0]*100)
# 畫圖
GOLD_INDEX_100.plot(figsize=(5,5))

# %%
# =============================================================================
# 2. 港股黃金股分析
# =============================================================================

# =============================================================================
# yahoo finance 拿取黃金期貨數據
# =============================================================================

Gold_Price = yf.Ticker('GC=F')
Gold_Price = Gold_Price.history(start='2020-01-01', end='2020-12-12')
Gold_Price = Gold_Price.Close
#因為時差﹐把整體數據往後移一日
Gold_Price_Shift = Gold_Price.shift(1)
# 改column 名字
Gold_Price.rename("Gold", inplace=True)
Gold_Price_Shift.rename("Gold_S", inplace=True)

#%%

# =============================================================================
# 富途拿取黃金股數據
# =============================================================================

quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
# 拿取所有板塊資料
ret, plate_data = quote_ctx.get_plate_list(Market.HK, Plate.CONCEPT)
# 搜索黃金板塊
plate_data = plate_data.loc[plate_data.plate_name.str.contains('黄金')]     
# 拿取所有包括在黃金股這板塊入面的股票
ret, stock_list = quote_ctx.get_plate_stock('HK.BK1222')
# 拿取所有包括在黃金股這板塊入面的股票名稱
stock_namelist = stock_list.stock_name
# 只要股票代碼
stock_list = stock_list.code.to_list()

# 設置一個vairable
data_set = []
for stock_code, stock_name in zip(stock_list, stock_namelist):
    # 拿取相關股票日K線    
    ret, data, page_req_key = quote_ctx.request_history_kline(stock_code, start='2020-01-01', 
                            end='2020-12-12', fields=[KL_FIELD.DATE_TIME, KL_FIELD.CLOSE])
    # 因為YahooFinance拿取的日期為DateTimeIndex﹐所以要把富途提取的time_key變為DateTimeIndex
    data.time_key = pd.to_datetime(data.time_key)
    # 設置日期作為index, 方便之後處理
    data = data.set_index('time_key')
    # 把column名close轉為股票名稱
    data.rename(columns={'close':stock_name}, inplace=True)
    # 把多餘的column code拿去
    data.drop(columns = 'code', inplace=True)
    # 把每隻股票數據放入data_set variable
    data_set.append(data)

# 把data_set由list轉為Dataframe    
data_set = pd.concat(data_set, axis=1)


#%%

# =============================================================================
# 畫heat map
# =============================================================================

# 把data_set和Gold_Price合拼
data_set = data_set.merge(Gold_Price, left_index=True, right_index=True)
# 把data_set和Gold_Price_Shift合拼
data_set = data_set.merge(Gold_Price_Shift, left_index=True, right_index=True)
# 以heatmap型式幫助分析
heatmap = sns.heatmap(data_set.pct_change().corr(), annot=True)

#%%

# =============================================================================
# 畫走勢圖
# =============================================================================

# 把沒有數據的日期拿走
data_set_rebase = data_set.dropna()
# 利用lamdba﹐把每一行的數據以x/x[0]*100公式運算一次
data_set_rebase = data_set_rebase.apply(lambda x: x/x[0]*100)
data_set_rebase.plot(figsize=(10,10))

quote_ctx.close()
