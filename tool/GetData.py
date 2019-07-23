#coding=utf-8

import pandas as pd
import numpy as np
import requests
import os

api = "https://api.insee.cn/tcm/prod/influxdb/query"  ## 生产环境的API

def Transfordata(res):
    if "series" not in res["results"][0]:
        return []
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


def GetNode_data(node_id = '',columns = ['quality', 'state', 'value'],factory = '',month = '',\
                 start = None,end = None,periods = None,freq=None,save = False):  
    ## 单变量 一次请求获取时间范围最大 为 3天, 多余三天的分段获取
    ## columns : 'quality', 'state', 'value'
    format = "%Y-%m-%d %H:%M:%S"
    times = pd.date_range(start= start,end = end,periods = periods,freq="3D")
    _start = times.min().strftime(format)
    _end = times.max().strftime(format)
    end1 = end if end else _end
    times = times.strftime(format).tolist()+[end1]
    times = list(set(times))
    times.sort()
    datas = []
    for index in range(len(times)-1):      
        res  =CalApi(node_id,times[index],times[index+1]) ## 中碳api接口调用
        temp = Transfordata(res)  ## 接口response 转换 dataframe
        if len(temp):
            datas.append(temp)
    if len(datas):
        df = pd.concat(datas)
        df = Decompression(df,end) ## 反解压 还原秒级数据
        ### 利用 quality 和 state 对数据进行过滤
        if freq:  ### 利用 freq 对数据进行 降采样
            df = df.resample(freq)[columns].mean().fillna(method = "ffill")
        
        if save:
            rawdir= '/'.join(['/home/khpython/Root/data/raw',factory,month])
            if df['value'].std()==0:
                print ("%s 值未发生变化 值为 %0.2f"%(node_id,df['value'].mean()))
            else:
                print (df.shape)
                if not os.path.exists(rawdir):
                    os.makedirs(rawdir)
                df.to_pickle(os.path.join(rawdir,"%s.pk"%node_id))
        return df
    else:
        return pd.DataFrame()

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

def ReadRawData(node_id = '',factory = "中碳能源",month = "2019_06",column = ['value']):
    rawdir= '/'.join(['/home/khpython/Root/data/raw',factory,month])
    path = os.path.join(rawdir,"%s.pk"%node_id)
    df = pd.read_pickle(path)
    if not len(column):
        column = df.columns
    return df[column]

def GenerateAvgTemp(nodes_id = ["TE_1003a", "TE_1003b", "TE_1003c"],column = "value",factory='中碳能源',\
                    month="2019_06",filename = None):
    dfs= []
    for node_id in nodes_id:
        df = ReadRawData(node_id = node_id,factory = factory,month = month,column = [column])
        if len(df):
            dfs.append(df)
    df = pd.concat(dfs,axis=1)
    df.columns = nodes_id
    df = df.fillna(method ="ffill")
    t = df.max(axis=1)
    df = df.apply(lambda x:x-t,axis=0)
    res = (df.sum(axis=1)+t*(len(df.columns)-1)/(len(df.columns)-1)).to_frame()
    res.columns = [column]
    if filename:
        path = os.path.join(rawdir,"%s.pk"%filename)
        res.to_pickle(path)
    return res

if __name__  == "__main__":
    ## 函数调用方式
    df = GetNode_data("TE_1003a",start = "2019-06-12 15:07:42",end = "2019-06-14 15:07:42")
    print (df.head())
    #
    df1 = GetNodes_data(["AT_1005","TE_1003a"],start = "2019-07-03 15:07:42",\
                        end =  "2019-07-05 17:07:42",freq="10min")
    print (df1.head())
