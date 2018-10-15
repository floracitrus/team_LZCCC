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
# clf_extra = ExtraTreesClassifier().fit(X,Y.values.ravel())
# model = SelectFromModel(clf_extra, prefit=True)
# X_new = model.transform(X)
#
# nb_features = X_new.shape[1]
# indices = np.argsort(clf_extra.feature_importances_)[::-1][:nb_features]
#
# X_newtrain = model.transform(X_train)
# X_newtest = model.transform(X_test)
# y_newtrain = Y.iloc[:math.ceil(0.75*len(X))]
# y_newtest = Y.iloc[math.ceil(0.75*len(X)):]
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


clf_x = XGBClassifier(max_depth = 6, gamma = 0.01)
clf_x1 = XGBClassifier(max_depth=3, eta = 0.1, gamma = 0, n_estimators=160)
clf_x2 = XGBClassifier(max_depth=5, n_estimators=300, eta=0.05)
clf_x3 = XGBClassifier(max_depth=4, eta = 0.08, n_estimators=150)
clf_x4 = XGBClassifier(max_depth=5, eta = 0.1, gamma = 0, min_child_weight = 6, n_estimators=150)


clf_r = clf_r.fit(X_train,y_train.values.ravel())
clf_rr = clf_rr.fit(X_train,y_train.values.ravel())
clf_rrr = clf_rrr.fit(X_train,y_train.values.ravel())
clf_g = clf_g.fit(X_train,y_train.values.ravel())
clf_d = clf_d.fit(X_train,y_train.values.ravel())

clf_x = clf_x.fit(X_train,y_train.values.ravel())
clf_x1 = clf_x1.fit(X_train,y_train.values.ravel())
clf_x2 = clf_x2.fit(X_train,y_train.values.ravel())
clf_x3 = clf_x3.fit(X_train,y_train.values.ravel())
clf_x4 = clf_x4.fit(X_train,y_train.values.ravel())

lo_model = lo_model.fit(X_train,y_train)
svm_model = svm.SVC(kernel = "linear",C=0.2)
svm_model = svm_model.fit(X_train,y_train)
svm_model1 = svm.SVC(kernel = "linear",C=0.4)
svm_model1 = svm_model1.fit(X_train,y_train)

print(classification_report(y_test,svm_model1.predict(X_test)))
print(classification_report(y_test,svm_model.predict(X_test)))
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

print("logistic regression")
print(classification_report(y_test,lo_model.predict(X_test)))



clf_r = clf_r.fit(X_newtrain,y_train.values.ravel())
clf_rr = clf_rr.fit(X_newtrain,y_train.values.ravel())
clf_rrr = clf_rrr.fit(X_newtrain,y_train.values.ravel())
clf_g = clf_g.fit(X_newtrain,y_train.values.ravel())
clf_d = clf_d.fit(X_newtrain,y_train.values.ravel())
clf_x = clf_x.fit(X_newtrain,y_train.values.ravel())
clf_xx = clf_xx.fit(X_newtrain,y_train.values.ravel())
clf_xxx = clf_xxx.fit(X_newtrain,y_train.values.ravel())
lo_model = lo_model.fit(X_newtrain,y_newtrain)

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

print("reduced features dimension xgboost with default setting")
print(classification_report(y_newtest.values.ravel(),clf_x.predict(X_newtest)))

print("reduced features dimension xgboost tuning max depth 3, nestimators = 300, learing rate 0.05")
print(classification_report(y_newtest.values.ravel(),clf_xx.predict(X_newtest)))

print("xgboost tuning max depth 8, etalearning_rate=0.1, nestimators = 100")
print(classification_report(y_newtest.values.ravel(),clf_xxx.predict(X_newtest)))
#
print("reduced features dimension logistic regression")
print(classification_report(y_newtest,lo_model.predict(X_newtest)))


print("finish train")
joblib.dump(clf_r, 'classifier_r.pkl')
joblib.dump(clf_rr, 'classifier_rr.pkl')
joblib.dump(clf_x, 'classifier_x.pkl')
joblib.dump(clf_x1, 'classifier_x1.pkl')
joblib.dump(clf_x2, 'classifier_x2.pkl')
