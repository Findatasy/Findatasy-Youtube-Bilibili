# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 21:58:47 2021

@author: Findatasy
"""

from futu import *
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import talib

#%%

# =============================================================================
# Part 1: draw daily money flow and price
# =============================================================================


#%%
# =============================================================================
# data download and import
# =============================================================================

# %%
# download daily capital flow data

quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)

ret, data = quote_ctx.get_capital_flow("HK.00700")
data.set_index('capital_flow_item_time', inplace=True)
data.index.rename('datetime', inplace=True)

data.drop(columns='last_valid_time', inplace=True)
if ret == RET_OK:
    print(data)
    
# download daily 1min kline
ret, stock_price, page_req_key = quote_ctx.request_history_kline('HK.00700', start='2021-02-05', 
end='2021-02-05', ktype=KLType.K_1M,fields=[KL_FIELD.DATE_TIME, KL_FIELD.CLOSE])  # 每页5个，请求第一页
stock_price.set_index('time_key', inplace=True)
stock_price.index.rename('datetime', inplace=True)
stock_price.drop(columns='code', inplace=True)
if ret == RET_OK:
    print(stock_price)
quote_ctx.close() # 结束后记得关闭当条连接，防止连接条数用尽
#%%
# save data

data.to_csv(r'C:\Users\chanl\Desktop\Futu_Lecture13\HK.01810data_0205.csv')
stock_price.to_csv(r'C:\Users\chanl\Desktop\Futu_Lecture13\HK.01810stock_price_0205.csv')


#%%
# import saved data

# day 1 
TC_Capitalflow_01 = pd.read_csv(r'D:\Dropbox\Findatasy\Futu_L13\data\HK.00700data_0104.csv', index_col=0)
TC_Stockprice_01 = pd.read_csv(r'D:\Dropbox\Findatasy\Futu_L13\data\HK.00700stock_price_0104.csv', index_col=0)

# day 2
TC_Capitalflow_02 = pd.read_csv(r'D:\Dropbox\Findatasy\Futu_L13\data\HK.00700data_0105.csv', index_col=0)
TC_Stockprice_02 = pd.read_csv(r'D:\Dropbox\Findatasy\Futu_L13\data\HK.00700stock_price_0105.csv', index_col=0)

# day 3
TC_Capitalflow_03 = pd.read_csv(r'D:\Dropbox\Findatasy\Futu_L13\data\HK.00700data_0106.csv', index_col=0)
TC_Stockprice_03 = pd.read_csv(r'D:\Dropbox\Findatasy\Futu_L13\data\HK.00700stock_price_0106.csv', index_col=0)

# day 4
TC_Capitalflow_04 = pd.read_csv(r'D:\Dropbox\Findatasy\Futu_L13\data\HK.00700data_0107.csv', index_col=0)
TC_Stockprice_04 = pd.read_csv(r'D:\Dropbox\Findatasy\Futu_L13\data\HK.00700stock_price_0107.csv', index_col=0)

#%%
# =============================================================================
# draw 2*2 subplots of stock price and money flow
# =============================================================================

#%%
# draw Tencent
# multiple subplots in one page
figTC, axTC = plt.subplots(2,2, figsize=(15,10))

# set primary axis
axTC1, axTC2 = axTC[0,:]
axTC3, axTC4 = axTC[1,:]

# set secondary axis
axTC1_s = axTC1.twinx()
axTC2_s = axTC2.twinx()
axTC3_s = axTC3.twinx()
axTC4_s = axTC4.twinx()

# draw on primary and secondary axis
TC_Stockprice_01.plot(ax=axTC1, color='green', legend=False)
TC_Capitalflow_01.plot(ax=axTC1_s, color='deepskyblue', legend=False)

TC_Stockprice_02.plot(ax=axTC2, color='green', legend=False)
TC_Capitalflow_02.plot(ax=axTC2_s, color='deepskyblue', legend=False)

TC_Stockprice_03.plot(ax=axTC3, color='green', legend=False)
TC_Capitalflow_03.plot(ax=axTC3_s, color='deepskyblue', legend=False)

TC_Stockprice_04.plot(ax=axTC4, color='green', legend=False)
TC_Capitalflow_04.plot(ax=axTC4_s, color='deepskyblue', legend=False)

# set x label of each subplots
axTC1.xaxis.label.set_visible(True)
axTC1.set_xticks([])
axTC1.set(xlabel="Day 1")

axTC2.xaxis.label.set_visible(True)
axTC2.set_xticks([])
axTC2.set(xlabel="Day 2")

axTC3.xaxis.label.set_visible(True)
axTC3.set_xticks([])
axTC3.set(xlabel="Day 3")

axTC4.xaxis.label.set_visible(True)
axTC4.set_xticks([])
axTC4.set(xlabel="Day 4")

# set master subtitle
figTC.suptitle("腾讯(700.HK)资金流向和股价", fontsize=30)

# get the labels for legend drawing
handles, labels = [(a + b) for a, b in zip(axTC1.get_legend_handles_labels(), axTC1_s.get_legend_handles_labels())]
# manually overide legend names, then draw master legend
labels[0] = "股价"
labels[1] = "资金流向"
figTC.legend(handles, labels, loc='upper right', fontsize=20)


# adjust layout
figTC.tight_layout()
# add margins around, then create master X Y labels
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.04)
figTC.text(0.5, 0.01, '交易日', va='center', ha='center', fontsize=15) # xlabel
figTC.text(0.03, 0.5, '股价', va='center', ha='center', rotation='vertical', fontsize=20) # ylabel left
figTC.text(0.97, 0.5, '资金流向', va='center', ha='center', rotation='vertical', fontsize=20) # ylabel right

#%%

# =============================================================================
# part 2. draw MFI and price
# =============================================================================

#%%
# download 5m, daily, weekly kline data
quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
ret, data, page_req_key = quote_ctx.request_history_kline('HK.00700', start='2021-01-04', end='2021-01-07', 
                                                          ktype=KLType.K_5M)  
data.to_csv(r'D:\Dropbox\Findatasy\Futu_L13\data\HK.00700_k5m.csv')

ret, data, page_req_key = quote_ctx.request_history_kline('HK.00700', start='2020-01-27', end='2021-01-26')  
data.to_csv(r'D:\Dropbox\Findatasy\Futu_L13\data\HK.00700_kday.csv')

ret, data, page_req_key = quote_ctx.request_history_kline('HK.00700', start='2016-01-27', end='2021-01-26', 
                                                          ktype=KLType.K_WEEK)  
data.to_csv(r'D:\Dropbox\Findatasy\Futu_L13\data\HK.00700_kweek.csv')
quote_ctx.close() # 结束后记得关闭当条连接，防止连接条数用尽


#%%

# =============================================================================
# draw MFI plots
# =============================================================================

#%%
# 腾讯5分钟k线数据，2021头4个交易日
TC_k5m = pd.read_csv(r'D:\Dropbox\Findatasy\Futu_L13\data\HK.00700_k5m.csv', index_col=0)

TC_k5m["MFI"] = talib.MFI(TC_k5m["high"], TC_k5m["low"], TC_k5m["close"], TC_k5m["volume"], timeperiod=14)
TC_k5m["Line20"] = 20
TC_k5m["Line80"] = 80

figTC_FMI, axTC_FMI = plt.subplots(figsize=(12,8))
axTC_FMI_s = axTC_FMI.twinx()
axTC_FMI.plot(TC_k5m['time_key'], TC_k5m['close'], color='green', label="股价")
###
axTC_FMI_s.plot(TC_k5m['time_key'], TC_k5m['MFI'], color='lightsalmon', label="MFI")
###
axTC_FMI_s.plot(TC_k5m['time_key'], TC_k5m['Line20'], color='lightsalmon', linestyle='dashed')
axTC_FMI_s.plot(TC_k5m['time_key'], TC_k5m['Line80'], color='lightsalmon', linestyle='dashed')
###
axTC_FMI.set_xticks([])
figTC_FMI.suptitle("腾讯(700.HK)股价和资金流量指标(MFI)", fontsize=20)
axTC_FMI.set_xlabel("2021年头4个交易日", fontsize=15)
axTC_FMI.set_ylabel("股价", fontsize=15)
axTC_FMI_s.set_ylabel("MFI", fontsize=15)
handles, labels = [(a + b) for a, b in zip(axTC_FMI.get_legend_handles_labels(), 
                                           axTC_FMI_s.get_legend_handles_labels())]
figTC_FMI.legend(handles, labels, loc="upper right")
figTC_FMI.tight_layout()



