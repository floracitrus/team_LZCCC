import pandas as pd
import os
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Ridge
from sklearn import svm
import math
import seaborn as sns
import matplotlib.pyplot as plt
stock_files = os.listdir("New_Stock_info")
Total = []
with open("A_stock") as f:
    A_list = eval(f.read())
A_set = set(A_list)
for stock_file in stock_files:
  if stock_file[0:6] in A_set:
    Data = pd.read_csv("New_Stock_info/"+stock_file,encoding="gbk")
    pct_list = list(Data["pct_change"])
    pct_list = pct_list[1:]+[0] #一周滞后
    Data["target"] = pd.Series(pct_list)
    Data["target_binary"] = Data["target"].apply(lambda x: 1 if x>0 else 0)
    Data["target_binary_2"] = Data["target"].apply(lambda x:1 if x>0.02 else 0)
    Data["target_binary_3"] = Data["target"].apply(lambda x: 1 if x > 0.05 else 0)
    Data["target_binary_4"] = Data["target"].apply(lambda x: 1 if x > 0.1 else 0)
    Data["target_binary_5"] = Data["target"].apply(lambda x: 1 if x > 0.2 else 0)

    pct_list = pct_list[1:] + [0]#二周滞后
    Data["target_2"] = pd.Series(pct_list)
    Data["target_2_binary"] = Data["target_2"].apply(lambda x: 1 if x > 0 else 0)
    Data["target_2_binary_2"] = Data["target"].apply(lambda x: 1 if x > 0.02 else 0)
    Data["target_2_binary_3"] = Data["target"].apply(lambda x: 1 if x > 0.05 else 0)
    Data["target_2_binary_4"] = Data["target"].apply(lambda x: 1 if x > 0.1 else 0)
    Data["target_2_binary_5"] = Data["target"].apply(lambda x: 1 if x > 0.2 else 0)


    pct_list = pct_list[1:] + [0]  # 三周滞后
    Data["target_3"] = pd.Series(pct_list)
    Data["target_3_binary"] = Data["target_3"].apply(lambda x: 1 if x > 0 else 0)
    Data["target_3_binary_2"] = Data["target"].apply(lambda x: 1 if x > 0.02 else 0)
    Data["target_3_binary_3"] = Data["target"].apply(lambda x: 1 if x > 0.05 else 0)
    Data["target_3_binary_4"] = Data["target"].apply(lambda x: 1 if x > 0.1 else 0)
    Data["target_3_binary_5"] = Data["target"].apply(lambda x: 1 if x > 0.2 else 0)

    stock_id = stock_file[:-4]
    stock_id = stock_id.replace("SH","XSHG")
    stock_id = stock_id.replace("SZ","XSHE")
    Data["stock_id"] = stock_id
    print(stock_id)
    Total.append(Data)
    del Data
Total_stock = pd.concat(Total)
del Total
Total_stock.fillna(method='ffill',inplace = True)
Total_stock.dropna(how="any",inplace =True)
Total_stock = Total_stock.replace(float('inf'),0)
Total_stock = Total_stock.replace(float('-inf'),0)
Total_columns = Total_stock.columns
non_indicator_column = ["date","open","close","high","low","code","pct_change","Unnamed: 0"]
y_header = ["target_binary_5"]
target_1 = ["target","target_binary","target_binary_2","target_binary_3","target_binary_4","target_binary_5"]
target_2 = ["target_2","target_2_binary","target_2_binary_2","target_2_binary_3","target_2_binary_4","target_binary_5"]
target_3 = ["target_3","target_3_binary","target_3_binary_2","target_3_binary_3","target_3_binary_4","target_binary_5"]

x_header = list(set(Total_columns).difference(set(non_indicator_column+target_1+target_2+target_3)))
m_columns = list(set(Total_columns).difference(set(non_indicator_column)))
Total_stock[m_columns].corr().to_csv("correlation_matrix.csv",encoding="gbk")
#for column in x_header:
    #Total_stock[column] = Total_stock[column].apply(lambda x:(x - Total_stock[column].mean())/Total_stock[column].std())
Total_stock.to_csv("Total_Stock_info.csv",encoding="gbk")
