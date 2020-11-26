"""
@author: Findatasy
"""

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
    stock_data = q.get()
    print(stock_data)
   
# =============================================================================
# Part 3
# =============================================================================
    
    for spot_price, high_price, stock_code in zip(stock_data['last_price'], stock_data['high_price'], stock_data['code']):
        #因為data的格式為DataFrame﹐需要把每行數據以zip的形式拿取作每行的分析
        
        if spot_price < high_price *0.95:
            # 如果相關個股現價比當日高位低過自設的百分比
           
            stock_position = int(portfolio_position['qty'][stock_code])
            # 提取相關個股的持倉股票數目
            if stock_position > 0:
            #如果相關的持倉股票數目 > 0
                
                trd_ctx.place_order(price= spot_price, qty=stock_position, code=stock_code, 
                                    order_type = OrderType.NORMAL, trd_side=TrdSide.SELL, trd_env = 'SIMULATE')
                #賣出相關股票
                #請留意因為是次教學是用模擬交易環境﹐所以用到Normal訂單﹐如果想直接賣出請轉做MARKET
                print('賣出 {}, {} 股'.format(stock_code, stock_position))
        else:
            # 如果相關個股現價比當日高位高於自設的百分比
            time.sleep(3)
            # 每3秒print一次未到價提醒
            print('{} 並未到價'.format(stock_code))

