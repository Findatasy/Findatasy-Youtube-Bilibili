"""
@author: Findatasy
"""

# =============================================================================
# futu doc
# https://openapi.futunn.com/futu-api-doc/quote/update-stock-quote.html
# https://openapi.futunn.com/futu-api-doc/trade/get-position-list.html#position-list-query
# https://openapi.futunn.com/futu-api-doc/quote/sub.html#subscribe
# =============================================================================

import queue
from futu import *

q = queue.Queue()

class StockQuoteTest(StockQuoteHandlerBase):
    def on_recv_rsp(self, rsp_str):
        ret_code, data = super(StockQuoteTest,self).on_recv_rsp(rsp_str)
        q.put(data) # 把數據放進q這 global variable
        return RET_OK, data

# 連接富途創建並初始化富途行情連接
quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111) 
# 創建相關交易市場並初始化相關市場交易連接
trd_ctx = OpenUSTradeContext(host='127.0.0.1', port=11111)
# 查詢持倉
ret_code, portfolio_position = trd_ctx.position_list_query(trd_env='SIMULATE')
# 把股票代碼變為Index
portfolio_position = portfolio_position.set_index('code')
# 设置实时报价回调
quote_ctx.set_handler(StockQuoteTest())  
# 订阅实时报价类型，FutuOpenD开始持续收到服务器的推送
quote_ctx.subscribe(portfolio_position.index.to_list(), [SubType.QUOTE]) 
while True:
    testing = q.get()
    print(testing)


