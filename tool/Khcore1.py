#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import requests
import os
from sklearn.linear_model import LinearRegression

basepath = 'D:/0.K2Data/4.AOI_Process/Kehang/data/raw'
# basepath = '/home/khpython/Root/data/raw'
#api = "https://api.insee.cn/tcm/prod/influxdb/query"  ## 生产环境的API
api = 'http://49.4.86.40:30009/tcm/prod/temp/influxdb/query'

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
    ## influxdb 存在一起sql 最大返回条目限制 10000条, 所以需要分段获取 2.5H频次获取数据
    times = pd.date_range(start= start,end = end,periods = periods,freq="2.5H")
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
            print (times[index],'---',times[index+1],":",temp.shape)
            datas.append(temp)
    if len(datas):
        df = pd.concat(datas)
        df = Decompression(df,end) ## 反解压 还原秒级数据
        ### 利用 quality 和 state 对数据进行过滤
        if freq:  ### 利用 freq 对数据进行 降采样
            df = df.resample(freq)[columns].mean().fillna(method = "ffill")
        
        if save:
            rawdir= '/'.join([basepath,factory,month])
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
    
    datas = []
    columns = []
    for node_id in node_ids:
        try:
            df = GetNode_data(node_id = node_id,columns = ['value'],\
                 start = start,end = end,periods = periods,save = False)
            ### 利用 quality 和 state 对数据进行过滤
            datas.append(df['value'])
            columns.append(node_id)
        except Exception as e:
            print ("%s get data error : %s "%(node_id,e))
            print (res)
    data = pd.concat(datas,axis=1)
    data = data.fillna(method = "ffill")
    data.columns = columns
    if freq:  ### 利用 freq 对数据进行 降采样
        data = data.resample(freq).mean().fillna(method = "ffill")
    return data

def ReadRawData(node_id = '',factory = "中碳能源",month = "2019_06",column = ['value']):

    rawdir= '/'.join([basepath,factory,month])
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
        rawdir= '/'.join([basepath,factory,month])
        path = os.path.join(rawdir,"%s.pk"%filename)
        res.to_pickle(path)
    return res



###########################################################################
def gradient(series):
    lr = LinearRegression()
    x = np.reshape(list(range(len(series))), (-1,1))
    line = lr.fit(x,series)
    
    return line.coef_[0]

#### function 1: 单变量某时间窗口的统计特征 ####
def generate_feature(series,column='value'): 
    df = pd.DataFrame(index = [series.index[-1]])
    df['Mean'] = series[column].mean()
    df['Std'] = series[column].std()
    df['Min'] = series[column].min()
    df['Max'] = series[column].max()
    df['Median'] = series[column].median()
    df['Q1'] = series[column].quantile(0.25)
    df['Q3'] = series[column].quantile(0.75)
    df['slope'] = gradient(series[column])
    if "label" in series.columns:
        df['label'] = series["label"].mean()
    return df


#### function 2: 单变量时序上的全部统计特征 ####
def SingleVarStat(node_id='',data = [],factory = "中碳能源",month = "2019_06" ,column = 'value', \
                  window = '30min', step = '5min', start=None, end =None, save = False):
    '''
    variale: str, 需要分析的变量名
    window: str，设置时间窗口大小，默认为30分钟
    step:str，设置步长，默认为5分钟
    start: str，设置起始时间，默认为None，取该变量本月第一个时间戳
    end：str，设置结束时间，默认为None，取该变量本月的最后一个时间戳
    save:bool，是否需要将数据存下来，默认为False
    '''
    rawdir= '/'.join([basepath,factory,month])
    savedir = rawdir.replace('/raw/','/process/')
    if len(data):
        df = data
    else:
        df = ReadRawData(node_id = node_id,factory = factory,month = month,column = [column])
        df = df.dropna().sort_index() ## 这个需要加, 让时间戳强制排顺
        print ("%s shape : %s"%(node_id,str(df.shape)))
    
    ## 使用 _start 和 _end 作为 传入date_range 的实参值 , 至于_start 和 _end 实际值是什么是由 start 和 end 决定的
    ## 你需要保证你的 _start 和 _end 为 Timestamp类型,因为df.index都是Timestamp类型
    _start = pd.to_datetime(start) if start else df.index[0]  
    _end = pd.to_datetime(end) if end else df.index[-1] 
    print ("start",_start,type(_start))
    print ("end",_end,type(_end))
    
    temp1 = pd.date_range(start=_start,end=_end -pd.to_timedelta(window),freq=step)
    temp2 = pd.date_range(start=_start+pd.to_timedelta(window),end=_end,freq=step)
    
    res = []
    for i in range(len(temp2)):
        series = df[(df.index>=temp1[i])&(df.index<temp2[i])]
        if len(series):
            feat = generate_feature(series,column = column)
            res.append(feat)
        
    dfnew = pd.concat(res)
    print ("%s feature : %s"%(node_id,str(dfnew.shape)))
    if save:
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        dfnew.to_pickle(os.path.join(savedir,"%s_%s_features.pk"%(node_id,window)))
   
    return dfnew


#### function 3: 根据时间区间赋值样本标签 ####
def generate_label(df,bins = [],labels=[]):
    '''
    df: 数据特征DataFrame, index 为 datetime类型
    bins : 时间范围区间 [] : 对应连续性 labels 赋值, [[]] : 对应非连续性 labels 赋值, 区间为 (]
    labels : 区间内对应的样本类别标签, 区间未覆盖到的标签为nan
    
    return 增加 label列的 df
    '''
    ## 以bins的第一个元素 来区分bins 是[] 还是 [[]]
    df1 = df.copy()
    if isinstance(bins[0],list):
        for index,(start,end) in enumerate(bins):
            df1.loc[df1[(df1.index>start)&(df1.index<=end)].index,"label"] = labels[index]
    else:
        bins = pd.to_datetime(bins)
        _labels = range(len(bins)-1)
        d = dict(zip(_labels,labels))
        df1['label'] = pd.cut(df1.index,bins = bins,labels=_labels)
        df1['label'] = df1['label'].map(d)
    return df1


#### function 4: 根据 node_id获取变量特征统计结果 ####
def GetSingleFeatures(node_id='',factory = "中碳能源",month = "2019_06" ,window ="30min",\
                      columns = ['Mean'] ,start = None,end = None):
    '''
    node_id : 变量的 node_id
    column : 获取node_id的 特征统计数据后, 筛选的字段名称
    start : 获取数据的起始时间
    end : 获取数据的截止时间
    '''
    dirname= '/'.join([basepath.replace('raw','process'),factory,month])
    path = os.path.join(dirname,"%s_%s_features.pk"%(node_id,window))
    df = pd.read_pickle(path)
    if start and end:  
        res = df[(df.index>=start)&(df.index<end)][columns]
    elif start:
        res = df[(df.index>=start)][columns]
    elif end:
        res = df[(df.index<end)][columns]
    else:
        res = df[columns]
    ###  需要补充根据 start 和 end 进行数据筛选的 逻辑
    return res


#### function 5: 根据 node_ids 组合变量特征统计结果 ####
def GetGroupFeatures(nodes_id = [],factory = "中碳能源",month = "2019_06" ,window ="30min", \
                    column = 'Mean',names = None,start = None,end = None):
    '''
    nodes_id : 待获取变量的 node_id列表
    column : 获取node_id的 特征统计数据后, 筛选的字段名称
    names :  自定义组合后数据的字段名列表, 默认为None, 则组合后数据字段名为 nodes_id
    start : 获取数据的起始时间
    end : 获取数据的截止时间
    '''
    datas = []
    for node_id in nodes_id:
        res = GetSingleFeatures(node_id,factory=factory,month=month,window=window,                            columns =[column] ,start = start,end = end)
        datas.append(res)
    df = pd.concat(datas,axis=1)
    if names:
        df.columns = names
    else:
        df.columns = nodes_id
    return df


def ReadSingleVarStat(node_id = '',factory = "中碳能源",month = "2019_06",window = "30min",column = []):
    rawdir= '/'.join([basepath.replace('raw','process'),factory,month])
    path = os.path.join(rawdir,"%s_%s_features.pk"%(node_id,window))
    df = pd.read_pickle(path)
    if not len(column):
        column = df.columns
    return df[column]


if __name__ == "__main__":
    df = SingleVarStat(node_id='AR-2201',factory = "中碳能源",month = "2019_06" ,column = 'value',\
                           window = '30min', step = '5min', save = True)
    dfs = GetGroupFeatures(nodes_id = ['AR-2201','AR-2202'],factory = "中碳能源",month = "2019_06" ,window ="30min", \
                            column = 'Mean',names = None,start = None,end = None)



