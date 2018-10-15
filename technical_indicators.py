#计算几类技术指标 KDJ，
import tushare
import os
import pandas as pd
from datetime import datetime
import numpy
stock_files = os.listdir("stock_price")
for stock_file in stock_files:
    code = stock_file.replace(".csv","")
    Data = pd.read_csv("stock_price/"+stock_file) #第1指标列为时间
    Data = pd.read_csv("000001.SZ.csv")
    #--------计算K,D,J指标----------#
    Data["RSV"] = (Data['close'] - Data['low'])/(Data['high'] - Data['low'])*100 #计算RSV
    K_list = []
    D_list = []
    J_list = []
    for i in range(0,len(Data)):
      if i == 0:
          K_list.append(50 * 2 / 3 + Data["RSV"][i] / 3)
          D_list.append(50 * 2 / 3 + K_list[i] / 3)
          J_list.append(3 * K_list[i] - 2 * D_list[i])
      else:
          K_list.append(K_list[i-1] * 2 / 3 + Data["RSV"][i] / 3)
          D_list.append(D_list[i-1] * 2 / 3 + K_list[i] / 3)
          J_list.append(3 * K_list[i] - 2 * D_list[i])
    Data["K"]=K_list
    Data["D"]=D_list
    Data["J"]=J_list
    #-------计算MACD---------------#
    EMA_12_list = []
    for i in range(0,len(Data)):
      if i == 0:
          EMA_12_list.append(11* Data["close"][0]/ 13 + 2* Data["close"][i]/ 13)
      else:
          EMA_12_list.append(11 * EMA_12_list[i-1] / 13 + 2 * Data["close"][i] / 13)
    Data["EMA12"] = EMA_12_list
    EMA_26_list = []
    for i in range(0,len(Data)):
      if i == 0:
          EMA_26_list.append(25* Data["close"][0]/ 27 + 2* Data["close"][i]/ 27)
      else:
          EMA_26_list.append(25 * EMA_26_list[i-1] / 27 + 2 * Data["close"][i] / 27)
    Data["EMA26"] = EMA_26_list
    Data["DIF"] = Data["EMA12"] - Data["EMA26"]
    DEA_list = []
    for i in range(0,len(Data)):
      if i == 0:
          DEA_list.append(8* Data["DIF"][0]/ 10 + 2* Data["DIF"][i]/ 10)
      else:
          DEA_list.append(8 * DEA_list[i-1] / 10 + 2 * Data["DIF"][i] / 10)
    Data["DEA"] = DEA_list
    Data["MACD"] = 2*(Data["DIF"]-Data["DEA"])
    #----------计算OBV------------#
    #OBA_list = []
    # time_base = "2016-11-10"##why
    # time_index = Data[(Data["date"]==time_base)].index
    # OBA_list.append(0)
    # for i in range(1,len(Data)):
    #   if Data["close"][i] > Data["close"][i-1]:
    #     OBA_list.append(Data["volume"][i] + Data["volume"])
    #   if Data["close"][i] < Data["close"][i-1]:
    #     OBA_list.append(Data["volume"] - Data["volume"][i])
    Data["OBA"] = ((Data['close']-Data['low'])-(Data['high']-Data['close']))/(Data['high']-Data['low'])*Data["volume"]
    #----------计算RSI------------#
    #相对强弱指标
    #国内单边做多的股市 : 强弱指标值一 般分布在 20 — 80; 80-100极强 卖出; 50-80强 买入; 20-50弱 观望; 0-20 极弱 买入。
    delta = Data["close"].diff()
    dUp, dDown = delta.copy(), delta.copy()
    dUp[dUp < 0] = 0
    dDown[dDown > 0] = 0
    #sliding window in 14
    #dUp
    Rolup = dUp.rolling(window=14,min_periods=1).mean()
    Roldown = dDown.rolling(window=14,min_periods=1).mean()
    RS = Rolup / Roldown
    RSI = 100*RS/(1+RS)
    Data['RSI'] = RSI

    #----------计算PSY------------#
    #PSY=N日内的上涨天数 /N×100%
    # N = 10#just assume, it doesn't say n = what...
    PSY = dUp.rolling(window=10,min_periods=1).apply(lambda x: numpy.count_nonzero((x>0)),raw = False)
    Data['PSY'] = PSY

    #----------计算AR------------#
    #人气指标 (AR)和意愿指标 (BR)两个指标构成
    #N日 AR=(N日内( H-O)之和除以 N日内( O-L)之和 )*100 其中，H为当日最高价，L 为当日最低价，O 为当日开盘价，N 为 设定的时间参数，一般原始参数日设定为 26 日
    #N = 26
    numo = (Data['high']-Data['open']).rolling(window=26,min_periods=1).sum()
    deno = (Data['open']-Data['low']).rolling(window=26,min_periods=1).sum()
    AR = numo/deno*100
    Data['AR'] = AR
    #N日 BR=N日内(H-CY)之和除以 N日内(CY-L)之和*100
    #C_y close price of previous day

    #----------计算BR------------#
    Cy = Data['close'].shift(1)
    Cy[0] = Cy[1] #处理因为是yesterday的close 导致的0 is NAN
    numo1 = (Data['high']-Cy).rolling(window=26,min_periods=1).sum()
    deno1 = (Cy-Data['low']).rolling(window=26,min_periods=1).sum()
    BR = numo1/deno1
    Data['BR'] = BR
    #----------计算CR------------#
    M = (2*Data['close']+Data['high']+Data['low'])/4
    Ym = M.shift(1)
    #Ym[0] = Ym[1] #处理因为是yesterday的 middle 导致的 0 index is NAN
    #N = 26
    P1 = (Data['high']-Ym).rolling(window = 26,min_periods=1).sum()
    P2 = (Ym-Data['low']).rolling(window = 26,min_periods=1).sum()
    CR = P1/P2*100
    CR[CR<0] = 0
    Data['CR'] = CR

    #----------计算BIAS------------#
    #BIAS=[( 当日收盘价 -N 日平均价 )/N 日平均价 ]*100%
    avg = (((Data['open']+Data['close'])/2).rolling(window = 6,min_periods=1).sum())/6
    BIAS_6 = (Data['close']-avg)/avg
    Data['BIAS_6'] = BIAS_6
    avg = (((Data['open']+Data['close'])/2).rolling(window = 12,min_periods=1).sum())/12
    BIAS_12 = (Data['close']-avg)/avg
    Data['BIAS_12'] = BIAS_12
    avg = (((Data['open']+Data['close'])/2).rolling(window = 24,min_periods=1).sum())/24
    BIAS_24 = (Data['close']-avg)/avg
    Data['BIAS_24'] = BIAS_24

    #----------计算CCI--------------#
    TP = (Data['high']+Data['low']+Data['close'])/3
    MA = Data['close'].rolling(window = 14,min_periods=1).sum()/14
    MD = (MA-Data['close']).rolling(window = 14,min_periods=1).sum()/14
    CCI = (TP - MA)/MD/0.015
    Data['CCI'] = CCI
    #----------计算WR---------------#
    n_days = 10
    lowest = Data['low'].rolling(min_periods=1, window=n_days).min()
    highest = Data['high'].rolling(min_periods=1, window=n_days).max()
    WR = 100*(highest-Data['close'])/(highest-lowest)
    Data['WR'] = WR
    #----------计算TRIX---------------#
    #1 TR=收盘价的 N 日指数移动平均值
    #2TRIX=(TR-昨日 TR)/昨日 TR*100
    #3MATRIX=TRIX的 M日简单移动平均
    #4参数 N设为 12，参数 M设为 20。

    TR = Data['close'].ewm(span=12, min_periods=1).mean()
    TRy = TR.shift(1)
    TRIX = (TR-TRy)/TRy*100
    MATRIX = TRIX.rolling(window=20, min_periods=1).mean()
    Data['M_TRIX']= MATRIX
