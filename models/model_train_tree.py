import math
import os
import pandas as pd
import seaborn as sns
import sys
import sklearn
import numpy as np
import pickle
from sklearn import svm
from sklearn.svm import LinearSVC
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
y_header = ["target_binary_5"]
target_1 = ["target","target_binary","target_binary_2","target_binary_3","target_binary_4","target_binary_5"]
target_2 = ["target_2","target_2_binary","target_2_binary_2","target_2_binary_3","target_2_binary_4","target_2_binary_5"]
target_3 = ["target_3","target_3_binary","target_3_binary_2","target_3_binary_3","target_3_binary_4","target_3_binary_5"]

x_header = list(set(Total_columns).difference(set(non_indicator_column+target_1+target_2+target_3)))
m_columns = list(set(Total_columns).difference(set(non_indicator_column)))
print(x_header)

X = Total_stock[x_header]
Y = Total_stock[y_header]
print("begin train")
X_train = X.iloc[:math.ceil(0.75*len(X))]
X_test = X.iloc[math.ceil(0.75*len(X)):]
y_train = Y.iloc[:math.ceil(0.75*len(X))]
y_test = Y.iloc[math.ceil(0.75*len(X)):]

#降维选取features
clf_extra = ExtraTreesClassifier().fit(X,Y.values.ravel())
model = SelectFromModel(clf_extra, prefit=True)
X_new = model.transform(X)

nb_features = X_new.shape[1]
indices = np.argsort(clf_extra.feature_importances_)[::-1][:nb_features]

X_newtrain = model.transform(X_train)
X_newtest = model.transform(X_test)
y_newtrain = Y.iloc[:math.ceil(0.75*len(X))]
y_newtest = Y.iloc[math.ceil(0.75*len(X)):]
# pca = PCA().fit(X)
# nb_features = X_new.shape[1]
# for f in range(nb_features):
# 	print("%d. feature %s (%f)" % (f + 1, x_header[1+indices[f]], clf_extra.feature_importances_[indices[f]]))

clf_r = sklearn.ensemble.RandomForestClassifier(n_estimators=100,max_depth = 7)
clf_rr = sklearn.ensemble.RandomForestClassifier(n_estimators = 60, min_samples_split = 4, max_depth = 5)
clf_rrr = sklearn.ensemble.RandomForestClassifier(n_estimators = 100, random_state=0, min_samples_split = 4, max_depth = 8)
clf_g = sklearn.ensemble.GradientBoostingClassifier(n_estimators=50)
clf_d = sklearn.tree.DecisionTreeClassifier(max_depth=10)
lo_model = LogisticRegression()

#grid search CV tuning最优参数
#parameters = [max_depth=3, eta = 0.1, gamma = 0, n_estimators=400]
#clf = grid_search.GridSearchCV(clf_x, parameters)
#clf.fit(X_train,y_train.values.ravel())
#tuning 不同的模型
# clf_x = XGBClassifier(max_depth = 6, gamma = 0.01)
# clf_x1 = XGBClassifier(max_depth=3, eta = 0.1, gamma = 0, n_estimators=160)
# clf_x2 = XGBClassifier(max_depth=5, n_estimators=300, eta=0.05)
# clf_x3 = XGBClassifier(max_depth=4, eta = 0.08, n_estimators=150)
# clf_x4 = XGBClassifier(max_depth=5, eta = 0.1, gamma = 0, min_child_weight = 6, n_estimators=150)


clf_r = clf_r.fit(X_train,y_train.values.ravel())
clf_rr = clf_rr.fit(X_train,y_train.values.ravel())
clf_rrr = clf_rrr.fit(X_train,y_train.values.ravel())
clf_g = clf_g.fit(X_train,y_train.values.ravel())
clf_d = clf_d.fit(X_train,y_train.values.ravel())

print("random forest limited nestimator = 50 max_depth = 7")
print(classification_report(y_test.values.ravel(),clf_r.predict(X_test)))
print("random forest limited n_estimators = 60, min_samples_split = 4, max_depth = 5")
print(classification_report(y_test.values.ravel(),clf_rr.predict(X_test)))
print("random forest limited n_estimators = 100, random_state=0, min_samples_split = 4, max_depth = 8")
print(classification_report(y_test.values.ravel(),clf_rrr.predict(X_test)))
print("gradient descent limited nestimator 50")
print(classification_report(y_test.values.ravel(),clf_g.predict(X_test)))
print("decision tree with depth 10")
print(classification_report(y_test.values.ravel(),clf_d.predict(X_test)))


clf_r = clf_r.fit(X_newtrain,y_train.values.ravel())
clf_rr = clf_rr.fit(X_newtrain,y_train.values.ravel())
clf_rrr = clf_rrr.fit(X_newtrain,y_train.values.ravel())
clf_g = clf_g.fit(X_newtrain,y_train.values.ravel())
clf_d = clf_d.fit(X_newtrain,y_train.values.ravel())


print("reduced features dimension random forest limited nestimator = 50 max_depth = 7")
print(classification_report(y_newtest.values.ravel(),clf_r.predict(X_newtest)))

print("reduced features dimension random forest limited n_estimators = 60, min_samples_split = 4, max_depth = 5")
print(classification_report(y_newtest.values.ravel(),clf_rr.predict(X_newtest)))

print("reduced features dimension random forest limited n_estimators = 100, random_state=0, min_samples_split = 4, max_depth = 8")
print(classification_report(y_newtest.values.ravel(),clf_rrr.predict(X_newtest)))

print("reduced features dimension radient descent limited nestimator 50")
print(classification_report(y_newtest.values.ravel(),clf_g.predict(X_newtest)))

print("reduced features dimension decision tree with depth 10")
print(classification_report(y_newtest.values.ravel(),clf_d.predict(X_newtest)))

print("finish train")
joblib.dump(clf_r, 'classifier_r.pkl')
joblib.dump(clf_rr, 'classifier_rr.pkl')
joblib.dump(clf_d, 'classifier_d.pkl')
