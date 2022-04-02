'''
@ WECHAT / TELEGRAM    :   MGSJZLS
@ Youtube Channel      :   youtube.com/c/美股数据张老师
'''

# 函数包含：单次获取400个标的的快照数据， 资金流 ， 大单 中单 小单 分布 
# 获取所有持仓 以及 亏损的持仓个股 
# 所有指标的Python代码史上最全 更多指标和策略代码请访问以上频道
'''
berration, above, above_value, accbands, ad, adosc, adx, alma, amat, ao, ao
bv, apo, aroon, atr, bbands, below, below_value, bias, bop, brar, cci, cdl_patte
rn, cdl_z, cfo, cg, chop, cksp, cmf, cmo, coppock, cross, cross_value, cti, deca
y, decreasing, dema, dm, donchian, dpo, ebsw, efi, ema, entropy, eom, er, eri, f
isher, fwma, ha, hilo, hl2, hlc3, hma, hwc, hwma, ichimoku, increasing, inertia,
jma, kama, kc, kdj, kst, kurtosis, kvo, linreg, log_return, long_run, macd, mad
, massi, mcgd, median, mfi, midpoint, midprice, mom, natr, nvi, obv, ohlc4, pdis
t, percent_return, pgo, ppo, psar, psl, pvi, pvo, pvol, pvr, pvt, pwma, qqe, qst
ick, quantile, rma, roc, rsi, rsx, rvgi, rvi, short_run, sinwma, skew, slope, sm
a, smi, squeeze, squeeze_pro, ssf, stc, stdev, stoch, stochrsi, supertrend, swma
, t3, td_seq, tema, thermo, tos_stdevall, trima, trix, true_range, tsi, tsignals
, ttm_trend, ui, uo, variance, vhf, vidya, vortex, vp, vwap, vwma, wcp, willr, w
ma, xsignals, zlma, zscore
'''
def get_snap(stock_list)
# stock_list 为富途支持的股票代码列表格式 最大支持400
	ret, data1 = quote_ctx.get_market_snapshot(stock_list[0:400])
	print(' k = '+str(k)+'  '+str(len(data1)))
	print(ret,len(data1),'获取快照')
	if ret==-1:
		print(data1)
	if ret!=-1 and len(data1)>1:
		data = data1.copy()
	elif ret==-1 :
		print(stock_list[0:400])
		print('\nError snapshot\n')
		print(data1)
		print(stock_list[0:400],len(stock_list[0:400]))
# 清空无效股票数据 价格0 或者 成交量0
# snap 没有涨跌幅，只能计算实时涨跌幅
	data = data.loc[~((data['last_price'] == 0) | (data['prev_close_price'] == 0))]
	data = data.loc[~((data['volume'] == 0) | (data['turnover'] == 0))]
	data['rate'] = data.apply(lambda x: (x['last_price']-x['prev_close_price']) / x['prev_close_price'] * 100, axis=1).round(decimals=2)
	data = data[['code','prev_close_price','last_price','rate','volume','turnover']].copy()
	df_m1 = df_m1.append(data[['code','last_price','volume','turnover','rate']])
	print('df_m1,data \t',len(df_m1),len(data))

def get_distribution(symbol):
#返回个股 资金分布 capital_in_mid 减去 capital_in_big 的差值
#中单 减去 大单 的差值
#可以任意发挥
	ret, data = quote_ctx.get_capital_distribution(symbol)
	if ret == RET_OK and len(data)>0:
		mid_big = data['capital_in_mid'][0] - data['capital_in_big'][0]
		return(mid_big)
	else:
		print('error:', data)
		return None
		
def get_mid(symbol, N):
#判断 capital_in_mid 大于 capital_in_big 和 capital_in_small
#判断 中单成交额 是否 大于 大单 以及 小单 
	ret, data = quote_ctx.get_capital_distribution(symbol)
	if ret == RET_OK:
		if data['capital_in_mid'].sum() < N:
			return False
		return(data['capital_in_mid'].sum() > max(data['capital_in_big'].sum(), data['capital_in_small'].sum()))
	else:
		return False
		
def get_flow(symbol, N, total):
# 获取 N 分钟内的资金流上升
	start = 0;		old_N = N - 1
	df_find = pd.DataFrame({})
	
	ret, data = quote_ctx.get_capital_flow(symbol)
	if ret != RET_OK:
		print(data)
		return False
	# print(data.tail(3))
	data['flow1'] = data['in_flow'].shift(-1) - data['in_flow'].copy()
	data.reset_index(drop=True, inplace=True)
	data['flow1'] = data['flow1'].shift(1)
	
	while N > 1:
		try:
			if data['flow1'].iloc[-1-start] > data['flow1'].iloc[-1-start-1]:
				start += 1
			else:				
				break
			N -= 1
		except:
			return False

	return(sum(data['in_flow'].values.tolist()) > N and start == old_N)
		
def get_query(TRD_ENV):
# 查询所有持仓 TRD_ENV 可以是 'REAL' / 'SIMULATE'
	df_move = pd.DataFrame({})
	ret,data_pos = trd_ctx.position_list_query(trd_env=TRD_ENV,refresh_cache=True)
	if ret==-1:
		def_txt('get_query position_list_query '+data_pos);		time.sleep(3.1)
	elif len(data_pos)>0:
		for i in reversed(range(len(data_pos))):	
			if data_pos['can_sell_qty'].iloc[i] > 0:
				try:
					df_move=df_move.append({'symbol':data_pos['code'].iloc[i],'cost_price':data_pos['cost_price'].iloc[i],'nominal_price':data_pos['nominal_price'].iloc[i],'symbol':data_pos['code'].iloc[i]},ignore_index=True)
				except:
					df_move = pd.DataFrame({'symbol':data_pos['code'].iloc[i],'cost_price':data_pos['cost_price'].iloc[i],'nominal_price':data_pos['nominal_price'].iloc[i],'symbol':data_pos['code'].iloc[i]},ignore_index=True)
	return(df_move)
df_move = get_query('REAL') #获取亏损的持仓个股
lose_num = 0
len_num = lose_num
if len(df_move)>len_num:
	df_lose = df_move[df_move['cost_price'] > df_move['nominal_price']].copy()
	print('\n',df_lose,'df_lose 获取亏损个股测试 \n ',df_lose)
	
import logging
import pandas as pd
import numpy as np
log = logging.getLogger(__name__)
		
def ma_x(df, n):
    """Calculate the moving average for the given data.
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    MA = pd.Series(df['close'].rolling(n, min_periods=n).mean(), name='MA_' + str(n))
    df = df.join(MA)
    return df


def exponential_moving_average(df, n):
    """
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    EMA = pd.Series(df['Close'].ewm(span=n, min_periods=n).mean(), name='EMA_' + str(n))
    df = df.join(EMA)
    return df


def momentum(df, n):
    """
    
    :param df: pandas.DataFrame 
    :param n: 
    :return: pandas.DataFrame
    """
    M = pd.Series(df['Close'].diff(n), name='Momentum_' + str(n))
    df = df.join(M)
    return df


def rate_of_change(df, n):
    """
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    M = df['Close'].diff(n - 1)
    N = df['Close'].shift(n - 1)
    ROC = pd.Series(M / N, name='ROC_' + str(n))
    df = df.join(ROC)
    return df


def average_true_range(df, n):
    """
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    i = 0
    TR_l = [0]
    while i < df.index[-1]:
        TR = max(df.loc[i + 1, 'high'], df.loc[i, 'Close']) - min(df.loc[i + 1, 'low'], df.loc[i, 'Close'])
        TR_l.append(TR)
        i = i + 1
    TR_s = pd.Series(TR_l)
    ATR = pd.Series(TR_s.ewm(span=n, min_periods=n).mean(), name='ATR_' + str(n))
    df = df.join(ATR)
    return df

def bollinger_bands(df, n):
    """
    
    :param df: pandas.DataFrame
    :param n:  millde is SMA 20
    :return: pandas.DataFrame
    """
    MA = pd.Series(df['close'].rolling(n, min_periods=n).mean())
    MSD = pd.Series(df['close'].rolling(n, min_periods=n).std())
    b1 = 4 * MSD / MA
    B1 = pd.Series(b1, name='BollingerB_' + str(n))
    df = df.join(B1)
    b2 = (df['close'] - MA + 2 * MSD) / (4 * MSD)
    B2 = pd.Series(b2, name='Bollinger%b_' + str(n))
    df = df.join(B2)
    return df

def boll(df):
	time_period = 25
	stdev_factor = 2  # 上下频带的标准偏差比例因子
	history = [];
	sma_values = []  # 初始化SMA值
	upper_band = []
	lower_band = []
	for close_price in df:
		history.append(close_price)
		if len(history) > time_period:
			del (history[0])
		sma = round(np.mean(history),2)
		sma_values.append(sma) 
		stdev = np.sqrt(np.sum((history - sma) ** 2) / len(history))  
		upper_band.append(round(sma + stdev_factor * stdev,2))
		lower_band.append(round(sma - stdev_factor * stdev,2))
	h_l_c=[]
	h_l_c.append(upper_band[-1]);h_l_c.append(sma_values[-1]);h_l_c.append(lower_band[-1])
	return h_l_c

def ppsr(df):
    """Calculate Pivot Points, Supports and Resistances for given data
    
    :param df: pandas.DataFrame
    :return: pandas.DataFrame
    """
    PP = pd.Series((df['high'] + df['low'] + df['Close']) / 3)
    R1 = pd.Series(2 * PP - df['low'])
    S1 = pd.Series(2 * PP - df['high'])
    R2 = pd.Series(PP + df['high'] - df['low'])
    S2 = pd.Series(PP - df['high'] + df['low'])
    R3 = pd.Series(df['high'] + 2 * (PP - df['low']))
    S3 = pd.Series(df['low'] - 2 * (df['high'] - PP))
    psr = {'PP': PP, 'R1': R1, 'S1': S1, 'R2': R2, 'S2': S2, 'R3': R3, 'S3': S3}
    PSR = pd.DataFrame(psr)
    df = df.join(PSR)
    return df


def stochastic_oscillator_k(df):
    """Calculate stochastic oscillator %K for given data.
    
    :param df: pandas.DataFrame
    :return: pandas.DataFrame
    """
    SOk = pd.Series((df['Close'] - df['low']) / (df['high'] - df['low']), name='SO%k')
    df = df.join(SOk)
    return df


def stochastic_oscillator_d(df, n):
    """Calculate stochastic oscillator %D for given data.
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    SOk = pd.Series((df['Close'] - df['low']) / (df['high'] - df['low']), name='SO%k')
    SOd = pd.Series(SOk.ewm(span=n, min_periods=n).mean(), name='SO%d_' + str(n))
    df = df.join(SOd)
    return df
#,N,M1,lower,upper)

def kdj(data):
	N=9;M1=3
	data['llv_low']=data['low'].rolling(N).min()
	data['hhv_high']=data['high'].rolling(N).max()
	data['rsv']=(data['close']-data['llv_low'])/(data['hhv_high']-data['llv_low'])
	data['k']=(data['rsv'].ewm(adjust=False,alpha=1/M1).mean())
	data['d']=(data['k'].ewm(adjust=False,alpha=1/M1).mean())
	data['j']=3*data['k']-2*data['d']


	return data
	
def trix(df, n):
    """Calculate TRIX for given data.
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    EX1 = df['Close'].ewm(span=n, min_periods=n).mean()
    EX2 = EX1.ewm(span=n, min_periods=n).mean()
    EX3 = EX2.ewm(span=n, min_periods=n).mean()
    i = 0
    ROC_l = [np.nan]
    while i + 1 <= df.index[-1]:
        ROC = (EX3[i + 1] - EX3[i]) / EX3[i]
        ROC_l.append(ROC)
        i = i + 1
    Trix = pd.Series(ROC_l, name='Trix_' + str(n))
    df = df.join(Trix)
    return df


def average_directional_movement_index(df, n, n_ADX):
    """Calculate the Average Directional Movement Index for given data.
    
    :param df: pandas.DataFrame
    :param n: 
    :param n_ADX: 
    :return: pandas.DataFrame
    """
    i = 0
    UpI = []
    DoI = []
    while i + 1 <= df.index[-1]:
        UpMove = df.loc[i + 1, 'high'] - df.loc[i, 'high']
        DoMove = df.loc[i, 'low'] - df.loc[i + 1, 'low']
        if UpMove > DoMove and UpMove > 0:
            UpD = UpMove
        else:
            UpD = 0
        UpI.append(UpD)
        if DoMove > UpMove and DoMove > 0:
            DoD = DoMove
        else:
            DoD = 0
        DoI.append(DoD)
        i = i + 1
    i = 0
    TR_l = [0]
    while i < df.index[-1]:
        TR = max(df.loc[i + 1, 'high'], df.loc[i, 'Close']) - min(df.loc[i + 1, 'low'], df.loc[i, 'Close'])
        TR_l.append(TR)
        i = i + 1
    TR_s = pd.Series(TR_l)
    ATR = pd.Series(TR_s.ewm(span=n, min_periods=n).mean())
    UpI = pd.Series(UpI)
    DoI = pd.Series(DoI)
    PosDI = pd.Series(UpI.ewm(span=n, min_periods=n).mean() / ATR)
    NegDI = pd.Series(DoI.ewm(span=n, min_periods=n).mean() / ATR)
    ADX = pd.Series((abs(PosDI - NegDI) / (PosDI + NegDI)).ewm(span=n_ADX, min_periods=n_ADX).mean(),
                    name='ADX_' + str(n) + '_' + str(n_ADX))
    df = df.join(ADX)
    return df

def myself_kdj(df):
    low_list = df['low'].rolling(9, min_periods=9).min()
    low_list.fillna(value=df['low'].expanding().min(), inplace=True)
    high_list = df['high'].rolling(9, min_periods=9).max()
    high_list.fillna(value = df['high'].expanding().max(), inplace=True)
    rsv = (df['close'] - low_list) / (high_list - low_list) * 100
    df['k'] = pd.DataFrame(rsv).ewm(com=2).mean()
    df['d'] = df['k'].ewm(com=2).mean()
    df['j'] = 3 * df['k'] - 2 * df['d']
    return df
def macdy(data,short=0,long1=0,mid=0):
    if short==0:
        short=12
    if long1==0:
        long1=26
    if mid==0:
        mid=9
    data['sema']=pd.Series(data['close']).ewm(span=short,min_periods = 11).mean()
    data['lema']=pd.Series(data['close']).ewm(span=long1,min_periods = 11).mean()
    data.fillna(0,inplace=True)
    data['DIF']=data['sema']-data['lema']
    data['DEA']=pd.Series(data['DIF']).ewm(span=mid, min_periods = 8).mean()
    data['MACD']=2*(data['DIF']-data['DEA'])
    data.fillna(0,inplace=True)
    return data

def macd(df, n_fast, n_slow):
    """Calculate MACD, MACD Signal and MACD difference
    :param df: pandas.DataFrame
    :param n_fast: 
    :param n_slow: 
    :return: pandas.DataFrame
    """
    EMAfast = pd.Series(df['Close'].ewm(span=n_fast, min_periods=n_slow).mean())
    EMAslow = pd.Series(df['Close'].ewm(span=n_slow, min_periods=n_slow).mean())
    MACD = pd.Series(EMAfast - EMAslow, name='MACD')
    MACDsign = pd.Series(MACD.ewm(span=9, min_periods=9).mean(), name='sign')
    MACDdiff = pd.Series(MACD - MACDsign, name='diff')
    df = df.join(MACD)
    df = df.join(MACDsign)
    df = df.join(MACDdiff)
    return df

def mass_index(df):
    """Calculate the Mass Index for given data.
    
    :param df: pandas.DataFrame
    :return: pandas.DataFrame
    """
    Range = df['high'] - df['low']
    EX1 = Range.ewm(span=9, min_periods=9).mean()
    EX2 = EX1.ewm(span=9, min_periods=9).mean()
    Mass = EX1 / EX2
    MassI = pd.Series(Mass.rolling(25).sum(), name='Mass Index')
    df = df.join(MassI)
    return df


def vortex_indicator(df, n):
    """Calculate the Vortex Indicator for given data.
    
    Vortex Indicator described here:

    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    i = 0
    TR = [0]
    while i < df.index[-1]:
        Range = max(df.loc[i + 1, 'high'], df.loc[i, 'Close']) - min(df.loc[i + 1, 'low'], df.loc[i, 'Close'])
        TR.append(Range)
        i = i + 1
    i = 0
    VM = [0]
    while i < df.index[-1]:
        Range = abs(df.loc[i + 1, 'high'] - df.loc[i, 'low']) - abs(df.loc[i + 1, 'low'] - df.loc[i, 'high'])
        VM.append(Range)
        i = i + 1
    VI = pd.Series(pd.Series(VM).rolling(n).sum() / pd.Series(TR).rolling(n).sum(), name='Vortex_' + str(n))
    df = df.join(VI)
    return df


def kst_oscillator(df, r1, r2, r3, r4, n1, n2, n3, n4):
    """Calculate KST Oscillator for given data.
    
    :param df: pandas.DataFrame
    :param r1: 
    :param r2: 
    :param r3: 
    :param r4: 
    :param n1: 
    :param n2: 
    :param n3: 
    :param n4: 
    :return: pandas.DataFrame
    """
    M = df['Close'].diff(r1 - 1)
    N = df['Close'].shift(r1 - 1)
    ROC1 = M / N
    M = df['Close'].diff(r2 - 1)
    N = df['Close'].shift(r2 - 1)
    ROC2 = M / N
    M = df['Close'].diff(r3 - 1)
    N = df['Close'].shift(r3 - 1)
    ROC3 = M / N
    M = df['Close'].diff(r4 - 1)
    N = df['Close'].shift(r4 - 1)
    ROC4 = M / N
    KST = pd.Series(
        ROC1.rolling(n1).sum() + ROC2.rolling(n2).sum() * 2 + ROC3.rolling(n3).sum() * 3 + ROC4.rolling(n4).sum() * 4,
        name='KST_' + str(r1) + '_' + str(r2) + '_' + str(r3) + '_' + str(r4) + '_' + str(n1) + '_' + str(
            n2) + '_' + str(n3) + '_' + str(n4))
    df = df.join(KST)
    return df


def rsi_x(df, n):
	"""Calculate Relative Strength Index(RSI) for given data.    :param df: pandas.DataFrame    :param n:     :return: pandas.DataFrame
	"""
	i = 0
	UpI = [0]
	DoI = [0]
	while i + 1 <= df.index[-1]:
		UpMove = df.loc[i + 1, 'high'] - df.loc[i, 'high']
		DoMove = df.loc[i, 'low'] - df.loc[i + 1, 'low']
		if UpMove > DoMove and UpMove > 0:
			UpD = UpMove
		else:
			UpD = 0
		UpI.append(UpD)
		if DoMove > UpMove and DoMove > 0:
			DoD = DoMove
		else:
			DoD = 0
		DoI.append(DoD)
		i = i + 1
	UpI = pd.Series(UpI)
	DoI = pd.Series(DoI)
	PosDI = pd.Series(UpI.ewm(span=n, min_periods=n).mean())
	NegDI = pd.Series(DoI.ewm(span=n, min_periods=n).mean())
	RSI = pd.Series(PosDI / (PosDI + NegDI), name='RSI_' + str(n))
	df = df.join(RSI)
	
	return df

def rsi2(df, periods = 24, ema = False):
    """
    Returns a pd.Series with the relative strength index.
    """
    close_delta = df['close'].diff()

    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    
    if ema == True:
        ma_up = up.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
        ma_down = down.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
    else:
        ma_up = up.rolling(window = periods).mean()
        ma_down = down.rolling(window = periods).mean()

    rsi = ma_up / ma_down
    rsi = 100 - (100/(1 + rsi))
    return rsi

def true_strength_index(df, r, s):
    """Calculate True Strength Index (TSI) for given data.
    
    :param df: pandas.DataFrame
    :param r: 
    :param s: 
    :return: pandas.DataFrame
    """
    M = pd.Series(df['Close'].diff(1))
    aM = abs(M)
    EMA1 = pd.Series(M.ewm(span=r, min_periods=r).mean())
    aEMA1 = pd.Series(aM.ewm(span=r, min_periods=r).mean())
    EMA2 = pd.Series(EMA1.ewm(span=s, min_periods=s).mean())
    aEMA2 = pd.Series(aEMA1.ewm(span=s, min_periods=s).mean())
    TSI = pd.Series(EMA2 / aEMA2, name='TSI_' + str(r) + '_' + str(s))
    df = df.join(TSI)
    return df


def accumulation_distribution(df, n):
    """Calculate Accumulation/Distribution for given data.
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    ad = (2 * df['Close'] - df['high'] - df['low']) / (df['high'] - df['low']) * df['Volume']
    M = ad.diff(n - 1)
    N = ad.shift(n - 1)
    ROC = M / N
    AD = pd.Series(ROC, name='Acc/Dist_ROC_' + str(n))
    df = df.join(AD)
    return df


def chaikin_oscillator(df):
    """Calculate Chaikin Oscillator for given data.
    
    :param df: pandas.DataFrame
    :return: pandas.DataFrame
    """
    ad = (2 * df['Close'] - df['high'] - df['low']) / (df['high'] - df['low']) * df['Volume']
    Chaikin = pd.Series(ad.ewm(span=3, min_periods=3).mean() - ad.ewm(span=10, min_periods=10).mean(), name='Chaikin')
    df = df.join(Chaikin)
    return df


def money_flow_index(df, n):
    """Calculate Money Flow Index and Ratio for given data.
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    PP = (df['high'] + df['low'] + df['Close']) / 3
    i = 0
    PosMF = [0]
    while i < df.index[-1]:
        if PP[i + 1] > PP[i]:
            PosMF.append(PP[i + 1] * df.loc[i + 1, 'Volume'])
        else:
            PosMF.append(0)
        i = i + 1
    PosMF = pd.Series(PosMF)
    TotMF = PP * df['Volume']
    MFR = pd.Series(PosMF / TotMF)
    MFI = pd.Series(MFR.rolling(n, min_periods=n).mean(), name='MFI_' + str(n))
    df = df.join(MFI)
    return df


def on_balance_volume(df, n):
    """Calculate On-Balance Volume for given data.
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    i = 0
    OBV = [0]
    while i < df.index[-1]:
        if df.loc[i + 1, 'Close'] - df.loc[i, 'Close'] > 0:
            OBV.append(df.loc[i + 1, 'Volume'])
        if df.loc[i + 1, 'Close'] - df.loc[i, 'Close'] == 0:
            OBV.append(0)
        if df.loc[i + 1, 'Close'] - df.loc[i, 'Close'] < 0:
            OBV.append(-df.loc[i + 1, 'Volume'])
        i = i + 1
    OBV = pd.Series(OBV)
    OBV_ma = pd.Series(OBV.rolling(n, min_periods=n).mean(), name='OBV_' + str(n))
    df = df.join(OBV_ma)
    return df


def force_index(df, n):
    """Calculate Force Index for given data.
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    F = pd.Series(df['Close'].diff(n) * df['Volume'].diff(n), name='Force_' + str(n))
    df = df.join(F)
    return df


def ease_of_movement(df, n):
    """Calculate Ease of Movement for given data.
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    EoM = (df['high'].diff(1) + df['low'].diff(1)) * (df['high'] - df['low']) / (2 * df['Volume'])
    Eom_ma = pd.Series(EoM.rolling(n, min_periods=n).mean(), name='EoM_' + str(n))
    df = df.join(Eom_ma)
    return df


def commodity_channel_index(df, n):
    """Calculate Commodity Channel Index for given data.
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    PP = (df['high'] + df['low'] + df['Close']) / 3
    CCI = pd.Series((PP - PP.rolling(n, min_periods=n).mean()) / PP.rolling(n, min_periods=n).std(),
                    name='CCI_' + str(n))
    df = df.join(CCI)
    return df


def coppock_curve(df, n):
    """Calculate Coppock Curve for given data.
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    M = df['Close'].diff(int(n * 11 / 10) - 1)
    N = df['Close'].shift(int(n * 11 / 10) - 1)
    ROC1 = M / N
    M = df['Close'].diff(int(n * 14 / 10) - 1)
    N = df['Close'].shift(int(n * 14 / 10) - 1)
    ROC2 = M / N
    Copp = pd.Series((ROC1 + ROC2).ewm(span=n, min_periods=n).mean(), name='Copp_' + str(n))
    df = df.join(Copp)
    return df


def keltner_channel(df, n):
    """Calculate Keltner Channel for given data.
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    KelChM = pd.Series(((df['high'] + df['low'] + df['Close']) / 3).rolling(n, min_periods=n).mean(),
                       name='KelChM_' + str(n))
    KelChU = pd.Series(((4 * df['high'] - 2 * df['low'] + df['Close']) / 3).rolling(n, min_periods=n).mean(),
                       name='KelChU_' + str(n))
    KelChD = pd.Series(((-2 * df['high'] + 4 * df['low'] + df['Close']) / 3).rolling(n, min_periods=n).mean(),
                       name='KelChD_' + str(n))
    df = df.join(KelChM)
    df = df.join(KelChU)
    df = df.join(KelChD)
    return df


def ultimate_oscillator(df):
    """Calculate Ultimate Oscillator for given data.
    
    :param df: pandas.DataFrame
    :return: pandas.DataFrame
    """
    i = 0
    TR_l = [0]
    BP_l = [0]
    while i < df.index[-1]:
        TR = max(df.loc[i + 1, 'high'], df.loc[i, 'Close']) - min(df.loc[i + 1, 'low'], df.loc[i, 'Close'])
        TR_l.append(TR)
        BP = df.loc[i + 1, 'Close'] - min(df.loc[i + 1, 'low'], df.loc[i, 'Close'])
        BP_l.append(BP)
        i = i + 1
    UltO = pd.Series((4 * pd.Series(BP_l).rolling(7).sum() / pd.Series(TR_l).rolling(7).sum()) + (
                2 * pd.Series(BP_l).rolling(14).sum() / pd.Series(TR_l).rolling(14).sum()) + (
                                 pd.Series(BP_l).rolling(28).sum() / pd.Series(TR_l).rolling(28).sum()),
                     name='Ultimate_Osc')
    df = df.join(UltO)
    return df


def donchian_channel(df, n):
    """Calculate donchian channel of given pandas data frame.
    :param df: pandas.DataFrame
    :param n:
    :return: pandas.DataFrame
    """
    i = 0
    dc_l = []
    while i < n - 1:
        dc_l.append(0)
        i += 1

    i = 0
    while i + n - 1 < df.index[-1]:
        dc = max(df['high'].ix[i:i + n - 1]) - min(df['low'].ix[i:i + n - 1])
        dc_l.append(dc)
        i += 1

    donchian_chan = pd.Series(dc_l, name='Donchian_' + str(n))
    donchian_chan = donchian_chan.shift(n - 1)
    return df.join(donchian_chan)


def standard_deviation(df, n):
    """Calculate Standard Deviation for given data.
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    df = df.join(pd.Series(df['Close'].rolling(n, min_periods=n).std(), name='STD_' + str(n)))
    return df
