with open("stock_graph", encoding="gbk") as file_object:
    stock_graph = eval(file_object.read())
with open("new_stock_graph",encoding="gbk") as file_object1:
    new_stock_graph = eval(file_object1.read())


count ={} # 计算各关系的条数
for code in stock_graph:
    for relation in stock_graph[code].keys():
        if relation in count.keys():
                count[relation] = count[relation] + len(stock_graph[code][relation])
        else:
            count[relation] = 0
            count[relation] = count[relation] + len(stock_graph[code][relation])
for code in new_stock_graph:
    for relation in new_stock_graph[code].keys():
        if relation in count.keys():

                count[relation] = count[relation] + len(new_stock_graph[code][relation])
        else:
            count[relation] = 0

            count[relation] = count[relation] + len(new_stock_graph[code][relation])
for relation in count.keys():
    count[relation] = count[relation]#/len(stock_graph.keys())
print(count)

with open("count","w",encoding="gbk") as f:
   f.write(str(count))
