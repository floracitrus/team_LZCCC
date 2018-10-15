#---寻找所有二级关系---#
with open("stock_graph", encoding="gbk") as file_object: #head_code为key的文件
    stock_graph = eval(file_object.read())
with open("stock_graph(reverse)",encoding="gbk") as file_object1:#target_name为key的文件
    stock_graph_reverse = eval(file_object1.read())

#基本想法：找出head_code的企业的关联组织，再用以target_name为key的文件中找出关联组织的关联上市A股企业（去掉回路）
new_stock_graph = {}
for head_key in stock_graph.keys():
    new_stock_graph[head_key] = {}
    for relation1 in stock_graph[head_key].keys():
        new_list = []
        for relevant in stock_graph[head_key][relation1]:
            try:
              if relevant["name"] in stock_graph_reverse.keys():
                name = relevant["name"]
                for relation2 in stock_graph_reverse[name].keys():
                    new_stock_graph[head_key][relation1+relation2] = list(set(stock_graph_reverse[name][relation2]))
                    new_stock_graph[head_key][relation1 + relation2].remove(head_key)
            except:
                continue

with open("new_stock_graph","w",encoding="gbk") as f:
   f.write(str(new_stock_graph))
