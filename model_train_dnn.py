import keras
from keras.layers import Dense,Dropout
from keras.optimizers import SGD
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import load_model
import numpy as np
Total_stock = pd.read_csv("Total_stock_info.csv",encoding="gbk")
Total_stock["target_binary_5"] = Total_stock["target"].apply(lambda x: 1 if x > 0.2 else 0)
Total_stock["target_binary_6"] = Total_stock["target"].apply(lambda x: 1 if x>  0.3 else 0)

Total_columns = Total_stock.columns
non_indicator_column = ["date","open","close","high","low","code","pct_change","Unnamed: 0","stock_id"]
y_header = ["target_binary_5"]
target_1 = ["target","target_binary","target_binary_2","target_binary_3","target_binary_4","target_binary_5","target_binary_6"]
target_2 = ["target_2","target_2_binary","target_2_binary_2","target_2_binary_3","target_2_binary_4"]
target_3 = ["target_3","target_3_binary","target_3_binary_2","target_3_binary_3","target_3_binary_4"]
x_header = list(set(Total_columns).difference(set(non_indicator_column+target_1+target_2+target_3)))
X = Total_stock[x_header]
Y = pd.concat([Total_stock[y_header],1-Total_stock[y_header]],axis=1)
print(Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

Model = keras.Sequential()
Model.add(Dense(200,activation='sigmoid',input_shape = (160,)))
Model.add(Dropout(0.5))
Model.add(Dense(100,activation="relu"))
Model.add(Dropout(0.5))
Model.add(Dense(160,activation="relu"))

Model.add(Dense(2,activation="softmax"))

sgd = SGD(lr=0.01,decay = 1e-6,momentum=0.9,nesterov=True)
Model.compile(loss="binary_crossentropy",optimizer=sgd)
print("begin train")
Model.fit(X_train,y_train,batch_size=200,epochs=50,shuffle=True,verbose=0,validation_split=0.3)
Model.save("neural_network.h5")
y_predict = np.around(Model.predict(X_test))
print(y_predict)
print(y_predict.shape)
print(y_test)
print(y_test.shape)
print(classification_report(y_test,y_predict))
