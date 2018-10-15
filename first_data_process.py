import pandas as pd
Data = pd.read_csv("Graph_Data.csv", encoding="gbk")
print(Data)
column_list = Data.columns # 获取指标列 column[0]为Head_code,column[1]为relation,column[2]为target_code,column[3]为target_name
print(len(Data))

first_head_code = ""
first_relation = ""
first_target_code = ""
first_target_name = ""
new_dic ={}
stock_graph ={}
for i in range(0,len(Data)):
    row_head_code = Data.iat[i,0]
    row_relation =  Data.iat[i,1]
    row_target_code = Data.iat[i,2]
    row_target_name = Data.iat[i,3]#取出一行的数据
    if(row_head_code!=first_head_code):#遇到新股票
        if (new_dic!={}):
            stock_graph[first_head_code]=new_dic
        first_head_code = row_head_code
        first_relation = row_relation
        new_dic = {}
        new_dic[row_relation] = []
        new_dic[row_relation].append({"code": row_target_code, "name": row_target_name})
    else:
        if(i==len(Data)-1):#结尾补充最后一支股票数据
            stock_graph[first_head_code] = new_dic
        if(first_relation!=row_relation):#遇到新关系
            first_relation = row_relation
            new_dic[row_relation] = []
            new_dic[row_relation].append({"code":row_target_code,"name":row_target_name})
        else:
            new_dic[row_relation].append({"code": row_target_code, "name": row_target_name})

with open("stock_graph","w",encoding="gbk") as f:
    f.write(str(stock_graph))
