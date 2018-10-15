import pandas as pd
import sys
import pickle
from sklearn.externals import joblib

clf_r = joblib.load('classifier5.pkl')
uniques_times = Total_stock[ 'date' ].unique()
uniques_times = sorted(uniques_times)
mylist = []
current_onhold = []
initval = 100000
count = 0
currval = 0

Total_stock= pd.read_csv("Total_Stock_info.csv",encoding="gbk")
Total_stock.sort_values(by = "date", ascending=True,inplace=True)

Total_columns = Total_stock.columns
non_indicator_column = ["date","open","close","high","low","code","pct_change","Unnamed: 0","stock_id"]
y_header = ["target_binary_5"]
target_1 = ["target","target_binary","target_binary_2","target_binary_3","target_binary_4","target_binary_5"]
target_2 = ["target_2","target_2_binary","target_2_binary_2","target_2_binary_3","target_2_binary_4","target_2_binary_5"]
target_3 = ["target_3","target_3_binary","target_3_binary_2","target_3_binary_3","target_3_binary_4","target_3_binary_5"]

x_header = list(set(Total_columns).difference(set(non_indicator_column+target_1+target_2+target_3)))
m_columns = list(set(Total_columns).difference(set(non_indicator_column)))
print(x_header)

X = Total_stock[x_header]
Y = Total_stock[y_header]
for t in uniques_times:
    #一个时间点是一个X_test
    X_test = X[Total_stock["date"] == t]
    #这个时间点的股票代码
    X_stockid = Total_stock[Total_stock["date"] == t].stock_id
    predictions = clf_r.predict_proba(X_test)
    predictions = np.c_[predictions,X_stockid]
    sorted_predict = sorted(predictions,key=lambda x: x[1])
    sorted_top_1 = sorted_predict[-1:]
    lst2 = [item[2] for item in sorted_top_1]
    lst2
    mylist.append({"date":t,"top5":lst2})

with open("mylist","w",encoding="gbk") as f:
    f.write(str(mylist))
