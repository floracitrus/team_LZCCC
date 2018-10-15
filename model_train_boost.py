import math
import os
import pandas as pd
import seaborn as sns
import sys
import sklearn
import numpy as np
import pickle
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from sklearn.decomposition import PCA


Total_stock= pd.read_csv("Total_Stock_info.csv",encoding="gbk")
Total_stock.sort_values(by = "date", ascending=True,inplace=True)

Total_columns = Total_stock.columns
non_indicator_column = ["date","open","close","high","low","code","pct_change","Unnamed: 0","stock_id"]
y_header = ["target_binary_4"]
target_1 = ["target","target_binary","target_binary_2","target_binary_3","target_binary_4"]
target_2 = ["target_2","target_2_binary","target_2_binary_2","target_2_binary_3","target_2_binary_4"]
target_3 = ["target_3","target_3_binary","target_3_binary_2","target_3_binary_3","target_3_binary_4"]
x_header = list(set(Total_columns).difference(set(non_indicator_column+target_1+target_2+target_3)))
m_columns = list(set(Total_columns).difference(set(non_indicator_column)))
print(x_header)
clf_x = XGBClassifier(max_depth = 6, gamma = 0.01)
clf_x1 = XGBClassifier(max_depth=3, eta = 0.1, gamma = 0, n_estimators=160)
clf_x2 = XGBClassifier(max_depth=5, n_estimators=300, eta=0.05)
clf_x3 = XGBClassifier(max_depth=4, eta = 0.08, n_estimators=150)
clf_x4 = XGBClassifier(max_depth=5, eta = 0.1, gamma = 0, min_child_weight = 6, n_estimators=150)




clf_x = clf_x.fit(X_train,y_train.values.ravel())
clf_x1 = clf_x1.fit(X_train,y_train.values.ravel())
clf_x2 = clf_x2.fit(X_train,y_train.values.ravel())
clf_x3 = clf_x3.fit(X_train,y_train.values.ravel())
clf_x4 = clf_x4.fit(X_train,y_train.values.ravel())







print("xgboost tuning max_depth = 6, gamma = 0.01 nestimator = 100, learning rate = 0.3")
print(classification_report(y_test.values.ravel(),clf_x.predict(X_test)))


print("xgboost tuning max_depth=3, eta = 0.1, gamma = 0, n_estimators=160")
print(classification_report(y_test.values.ravel(),clf_x1.predict(X_test)))

print("xgboost tuning max_depth=5, n_estimators=300, eta=0.05")
print(classification_report(y_test.values.ravel(),clf_x2.predict(X_test)))


print("xgboost tuning max_depth=4, eta = 0.08, n_estimators=150")
print(classification_report(y_test.values.ravel(),clf_x3.predict(X_test)))

print("xgboost tuning max_depth=5, eta = 0.1, gamma = 0, min_child_weight = 6, n_estimators=150")
print(classification_report(y_test.values.ravel(),clf_x4.predict(X_test)))



y = clf_x.predict(X_test)
y1 = clf_x1.predict(X_test)
y2 = clf_x2.predict(X_test)
y3 = clf_x3.predict(X_test)
y4 = clf_x4.predict(X_test)
fold = (y+y1+y2+y3+y4)/5
