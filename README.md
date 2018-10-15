# team_LZCCC 
利用企业知识图谱与机器学习智能选择最优投资组合

Select the best bucket of stocks by Machine Learning and Enterprise Knowledge Graph

本研究策略框架如图2所示，先是用网络爬虫和自然语言新闻挖掘等手段整合企业关联数据，从而构建以全体A股上市公司为核心的企业知识图谱。而后本研究又从Tushare(股票公开数据源)提取出股票数据，与企业知识图谱数据相结合计算出各类可能存在超额收益的关联企业影响力因子。而后我们将数据放入多类机器学习模型中进行训练，挑出在测试集上表现最佳的模型。最后根据该模型预测出下一周股票收益正概率的排序，选出排序较前的股票进行持仓。

![image](http://github.com/team_LZCCC/readme_add_pic/raw/master/回测表现.jpg)
