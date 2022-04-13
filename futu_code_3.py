'''
Designed by : MGSJZLS (wechat / telegram)
Channel 	: youtube.com/c/美股数据张老师
更多更全代码：请上频道留言
'''

from futu import *

def ftw(min1,max1,min2,max2):# 此函数自定义牛熊最小最大价格 按照成交额进行搜号写盘
	req=Request()
	req.sort_field=SortField.TURNOVER
	req.status=WarrantStatus.NORMAL

	req.cur_price_min=float(min1)
	req.cur_price_max=float(max1)
	print(float(min1),float(max1))

	# req.issuer_list = ['BI','HT','HS','CS','UB','BP','SG','VT'] # less BI HS UB 
	req.conversion_min = 10000
	req.conversion_max = 10000

	f=open(r'ftw.csv','w');f.write('');f.close()	
	csvFile = open('ftw.csv','a+', newline='')
	write_ok = csv.writer(csvFile)
###
	req.type_list=['BULL']
	try:
		ret, ls = quote_ctx.get_warrant("HK.800000", req)
		if ret == RET_OK:
			data, last_page, all_count = ls	
		else:
			print('\n 此价格区间 无法搜索到牛熊证号码 \n')
			def_txt('ftw try error')
	except:
		req.cur_price_min=0.040
		req.cur_price_max=0.2
		ret , (data, last_page, all_count) = quote_ctx.get_warrant("HK.800000", req)

	if ret!=-1 and len(data)>0:# lambda x:'%.2f' % x
		data['turnover'] =data.apply(lambda x: int(x['turnover'] / 10000), axis=1)
		for i in data[['stock','recovery_price','cur_price','turnover','type','issuer']].iloc[-8:].values:
			write_ok.writerow(i)	
		# print(data[['stock','recovery_price','cur_price','turnover','type','issuer']])
			
	elif ret!=-1 and len(data)==0:	
		req.cur_price_min=0.040
		req.cur_price_max=0.2
		req.type_list=['BULL']
		ret , (data, last_page, all_count) = quote_ctx.get_warrant("HK.800000", req)
		data['turnover'] =data.apply(lambda x: int(x['turnover'] / 10000), axis=1)
		for i in data[['stock','recovery_price','cur_price','turnover','type','issuer']].iloc[-8:].values:
			write_ok.writerow(i)		
	
	req.cur_price_min=float(min2)
	req.cur_price_max=float(max2)
	print(float(min2),float(max2))
		
	req.type_list=['BEAR']
	
	try:
		ret, ls = quote_ctx.get_warrant("HK.800000", req)
		if ret == RET_OK:
			data, last_page, all_count = ls	
		else:
			print(ls)
	except:
		req.cur_price_min=0.040
		req.cur_price_max=0.2
		ret , (data, last_page, all_count) = quote_ctx.get_warrant("HK.800000", req)
	
	if ret!=-1 and len(data)>0:
		data['turnover'] =data.apply(lambda x: int(x['turnover'] / 10000), axis=1)#issuer #conversion_ratio #name #street_rate #issue_size
		for i in data[['stock','recovery_price','cur_price','turnover','type','issuer']].iloc[-8:].values:
			write_ok.writerow(i)
		# print(data[['stock','recovery_price','cur_price','turnover','type','issuer']])	
			
	elif ret!=-1 and len(data)==0:	
		print(data)
		req.cur_price_min=0.040
		req.cur_price_max=0.250
		req.type_list=['BEAR']
		ret , (data, last_page, all_count) = quote_ctx.get_warrant("HK.800000", req)
		data['turnover'] =data.apply(lambda x: int(x['turnover'] / 10000), axis=1)#issuer #conversion_ratio #name #street_rate #issue_size
		for i in data[['stock','recovery_price','cur_price','turnover','type','issuer']].iloc[-8:].values:
			write_ok.writerow(i)
			
def place_(symbol, price, cash, side, volume ,TRD_ENV):# 此函数为通用下单代码,TRD_ENV:'REAL' / 'SIMULATE'
	trd_side = TrdSide.SELL if side == 'sell' else TrdSide.BUY			
	ret,data_err=trd_ctx.place_order(price=0, qty=volume, code=symbol, trd_side=trd_side,  order_type=OrderType.MARKET, trd_env=TRD_ENV)
	return(data_err)
	
if __name__ == '__main__':
	quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
	trd_ctx = OpenHKTradeContext(host='127.0.0.1', port=11111,is_encrypt = False, security_firm=SecurityFirm.FUTUSECURITIES)
	read_file = open(r'pwd.txt', 'r');	 # pwd.txt 存入六位数富途交易密码
	print('\n',quote_ctx,'\n',trd_ctx)
	pwd_unlock='';selling=0
	if len(pwd_unlock)>1:
		print('\n',trd_ctx.unlock_trade(pwd_unlock),'\n')
	ftw(0.05， 0.12， 0.06， 0.13) # 调用函数 传入 牛熊最小最大价格 进行搜号写盘