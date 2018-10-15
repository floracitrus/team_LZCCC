import os
import pandas as pd
import numpy as np
from collections import Counter
stock_files = os.listdir("stock_price")

with open("stock_graph3",encoding="gbk") as f:
    stock_graph = eval(f.read())
with open("new_stock_graph",encoding="gbk") as f:
    stock_graph_2 = eval(f.read())
with open("count",encoding="gbk") as f:
    count = eval(f.read())

#---将两个字典合并---#
dic={}
for code in stock_graph.keys():
    dic[code] = {**stock_graph[code], **stock_graph_2[code]}
stock_graph = dic
print(stock_graph.keys())


Total_stock = {}
basic_column  =["open","close","high","low","volume","code","pct_change"]
for stock in stock_files:
    try:
        code = stock[0:9]
        Total_stock[code] = pd.read_csv("stock_price/"+stock).set_index("date")
        Total_stock[code].rename(columns=lambda x: x+code,inplace = True)
    except:
        print("fail",stock)
for code in Total_stock.keys():
  print(code)
  try:
    relevants = []
    for relation in stock_graph[code].keys():
        relevants = relevants + stock_graph[code][relation]
    relevants.append(code)
    relevants = list(set(relevants))
    Total = pd.concat([Total_stock[relevant] for relevant in relevants], join_axes=[Total_stock[code].index],axis=1)
    Total.fillna(0,inplace=True)
    for relation in count.keys():
        Total[relation+code] = 0
    relevants.remove(code)
    for relation in stock_graph[code].keys():
        relevants = stock_graph[code][relation]
        list_sum = np.sum([Total["pct_change"+relevant] for relevant in relevants],axis=0)
        Total[relation+code] = list_sum
    indicator_list = basic_column + list(count.keys())


    Data = Total[list(map(lambda x:x+code,indicator_list))]

    Data.rename(columns = lambda x:x.replace(code,""),inplace =True)
    Data.to_csv("new_stock_price/"+code+".csv")
  except:
      print(code,"not work")


