# -*- coding: utf-8 -*-
"""
@author: Findatasy
"""


from futu import *
import pandas as pd
import seaborn as sns
# pip install seaborn

#%%
# =============================================================================
# 方法1：根据目前持仓股票
# =============================================================================

quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
# 連接富途創建並初始化富途行情連接
trd_ctx = OpenHKTradeContext(host='127.0.0.1', port=11111) 
# 創建相關交易市場並初始化相關市場交易連接

ret, position_list = trd_ctx.position_list_query(trd_env='SIMULATE')
# 拿取股票持倉資料
position_list = position_list.code.to_list()
# 只拿取股票代碼
position_list.append('HK.800000')
# 加入恆指代碼
quote_ctx.close()
trd_ctx.close()


#%%
# =============================================================================
# 方法2：自定义股票
# =============================================================================

position_list = ['HK.09988', 'HK.01810', 'HK.00941', 'HK.00175', 'HK.00853', 'HK.800000']
# 'HK.09988' BABA, 'HK.01810' XIAOMI, 'HK.00941' CHINA MOBILE, 
# 'HK.00175' GEELY AUTO, 'HK.00853' MICROPORT, 'HK.800000' HSI


#%%

data_set = []
# 設置一個vairable

quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
# 連接富途創建並初始化富途行情連接

for stock_code in position_list:
# for loop股票代碼
    ret, data, page_req_key = quote_ctx.request_history_kline(
        stock_code, start='2020-01-01', end='2020-11-30', 
        fields=[KL_FIELD.DATE_TIME, KL_FIELD.CLOSE])
    # 拿取相關股票日K線
    
    data = data.set_index('time_key')
    # 設置日期作為index, 方便之後處理
    data.rename(columns={'close':stock_code}, inplace=True)
    # 把column名close轉為股票代碼
    data.drop(columns = 'code', inplace=True)
    # 把多餘的column拿去
    data_set.append(data)
    # 把每隻股票數據放入data_set variable

data_set = pd.concat(data_set, axis=1)
# 把data_set由list轉為Dataframe
data_set = data_set.pct_change()
# 計算每1個股票的日轉變
heatmap = sns.heatmap(data_set.corr(), annot=True)
# 以heatmap型式幫助分析





    
    

