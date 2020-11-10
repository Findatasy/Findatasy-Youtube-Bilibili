#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from futu import *
pwd_unlock = '123456' #填寫返自己的交易解鎖密碼
trd_ctx = OpenUSTradeContext(host='127.0.0.1', port=11111) #如果是不同交易市場相關初始化交易連接會不同
print(trd_ctx.unlock_trade(pwd_unlock))
print(trd_ctx.place_order(price=420, qty=100, code="US.TSLA", trd_side=TrdSide.SELL))
trd_ctx.close()


# In[ ]:





# In[ ]:




