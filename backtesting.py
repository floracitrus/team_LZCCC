# 可以自己import我们平台支持的第三方python模块，比如pandas、numpy等。
from rqalpha.api import *
from datetime import date
from datetime import time
from datetime import datetime
from datetime import timedelta

# 在这个方法中编写任何的初始化逻辑。context对象将会在你的算法策略的任何方法之间做传递。
with open("mylist") as f:
    mylist = eval(f.read())


mydict={
    i['date']:i['top5'] for i in mylist
    }

def init(context):
    # 在context中保存全局变量
    #context.s1 = "000001.XSHE"
    # 实时打印日志
    context.count = 0
    logger.info("RunInfo: {}".format(context.run_info))


# before_trading此函数会在每天策略交易开始前被调用，当天只会被调用一次
def before_trading(context):
    pass


# 你选择的证券的数据更新将会触发此段逻辑，例如日或分钟历史数据切片或者是实时数据切片更新
def handle_bar(context, bar_dict):
    # 开始编写你的主要的算法逻辑

    # bar_dict[order_book_id] 可以拿到某个证券的bar信息
    # context.portfolio 可以拿到现在的投资组合信息

    # 使用order_shares(id_or_ins, amount)方法进行落单
    #if(mylist["date"] in context.now)
    #if 这个context.now == 现在的date了
    #就进行买
    currdate = context.now.strftime("%Y-%m-%d")
    currwday = context.now.strftime("%A")
    lastweek = (context.now - timedelta(days=7)).strftime("%Y-%m-%d")
    #print(lastweek)
    if currwday == "Friday":
        try:
            if context.count>0:
                lastweek_share = mydict[lastweek]
                for l in range(len(lastweek_share)):
                   try:
                    order_shares(str(lastweek_share[l]), -1000)
                    print("selling stock"+str(lastweek_share[l]))
                   except:
                       print("not this stock")
                logger.info(lastweek_share)

            today_share = mydict[currdate]
            for s in range(len(today_share)):
              try:
                order_shares(str(today_share[s]),1000)
                print("buying stock"+str(today_share[s]))
              except:
                  print("CAN'T BUT B:"+today_share[s])
            logger.info(today_share)



            context.count+=1
        except:
            print("time wrong")

# after_trading函数会在每天交易结束后被调用，当天只会被调用一次
def after_trading(context):
    pass

# run_file_demo
from rqalpha import run_file

config = {
  "base": {
    "data_bundle_path":"E:\\Strategy\\bundle",
    "start_date": "2012-01-01",
    "end_date": "2018-01-01",
    "benchmark": "000300.XSHG",
    "accounts": {
        "stock": 100000
    }
  },
  "extra": {
    "log_level": "verbose",
  },
  "mod": {
    "sys_analyser": {
      "enabled": True,
      "plot": True
    }
  }
}

strategy_file_path = "backtest.py"

run_file(strategy_file_path, config)

