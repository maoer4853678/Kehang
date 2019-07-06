#coding=utf-8

import pandas as pd
import numpy as np
import requests

api = "https://api.insee.cn/tcm/prod/influxdb/query"  ## 生产环境的API

def Transfordata(res):
    columns = res["results"][0]['series'][0]['columns']
    data = res["results"][0]['series'][0]['values']
    df = pd.DataFrame(data,columns = columns)
    df['time'] = pd.to_datetime(df['time'].map(lambda x:x.split("+")[0]))
    df = df.sort_values('time')
    df = df.set_index("time")
    df = df.drop("node_id",axis=1)
    return df

def Decompression(df,end):
    if df.index.max()!= pd.to_datetime(end):        
        df.loc[pd.to_datetime(end)] = np.nan
    df = df.resample("1S").fillna(method = "ffill").dropna()
    return df

def CalApi(node_id,start,end):
    q = '''select * from data where node_id='%s' and time >= '%s' and time < '%s' tz('Asia/Shanghai'); '''%\
        (node_id,start,end)
    r = requests.post(api, data = {"db":"kehang","q":q},verify=False)
    return r.json()


def GetNode_data(node_id = '',start = None,end = None,periods = None,freq=None):
    times = pd.date_range(start= start,end = end,periods = periods,freq=None)
    start = times.min()
    end = times.max()
    res  =CalApi(node_id,start,end) ## 中碳api接口调用
    df = Transfordata(res)  ## 接口response 转换 dataframe
    df = Decompression(df,end) ## 反解压 还原秒级数据
    
    ### 利用 quality 和 state 对数据进行过滤
    if freq:  ### 利用 freq 对数据进行 降采样
        df = df.resample(freq).mean().fillna(method = "ffill")
    return df

def GetNodes_data(node_ids = [],start = None,end = None,periods = None,freq=None):
    '''为了防止 接口频繁调用给生产数据库带来压力，调用GetNodes_data 时，一次调用时，node_ids个数控制在50以内
     时间范围控制在 2天以内
    '''
    times = pd.date_range(start= start,end = end,periods = periods,freq=None)
    start = times.min()
    end = times.max()
    datas = []
    columns = []
    for node_id in node_ids:
        try:
            res = CalApi(node_id,start,end)
            df = Transfordata(res)
            ### 利用 quality 和 state 对数据进行过滤
            datas.append(df['value'])
            columns.append(node_id)
        except Exception as e:
            print ("%s get data error : %s "%(node_id,e))
            print (res)
    data = pd.concat(datas,axis=1)
    data.columns = columns
    data = Decompression(data,end)
    if freq:  ### 利用 freq 对数据进行 降采样
        data = data.resample(freq).mean().fillna(method = "ffill")
    return data

if __name__  == "__main__":
    ## 函数调用方式
    df = GetNode_data("TE_1003a",start = "2019-06-12 15:07:42",end = "2019-06-14 15:07:42")
    print (df.head())
    #
    df1 = GetNodes_data(["AT_1005","TE_1003a"],start = "2019-07-03 15:07:42",\
                        end =  "2019-07-05 17:07:42",freq="10min")
    print (df1.head())
