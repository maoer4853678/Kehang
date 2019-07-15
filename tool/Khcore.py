#coding=utf-8
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler as _MinMaxScaler
import zipfile
import configparser
import functools
import time
import math
import json
import numpy as np
import pylab as pl
from pylab import mpl
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

## 各项配置参数的绝对路径

RAWDATA = '/home/khpython/Root/data/raw' 
INIT = '/home/khpython/Root/init' 

## 函数执行耗时统计装饰器
def costtime(func):
    @functools.wraps(func)
    def wrapper(*args,**args1):
        st = time.time()
        res = func(*args,**args1)
        et = time.time()
        print ("%s函数 执行耗时: %0.2fs"%(func.__name__,et-st))
        return res
    return wrapper

def UnzipFiles(path):
    z = zipfile.ZipFile(path)
    z.namelist()
    for f in z.namelist():
        z.extract(f)

def _getconf():
    conf=configparser.ConfigParser()
    conf.read(os.path.join(INIT,"init.ini"))    ## init.ini 管理着 factory 对应的init文件地址，通过更新init文件映射关系即可
    return conf

def GetPeriods(factory = "吉林大成",month='2018_09'):
    conf =_getconf()
    periods = {}
    if "%s_periods"%factory in conf.sections():
        ## 检查该factory 是否存在周期配置项
        if month in list(conf["%s_periods"%factory].keys()):
            ## 检出周期配置项 是否存在 输入月份
            periods = json.loads(conf["%s_periods"%factory][month])
        else:
            print ("%s周期配置中未发现%s , 当前周期配置项目如下:"%(factory,month))
            print (list(conf["%s_periods"%factory].keys()))
    else:
        print ("%s不存在周期配置 , 当前周期配置项目如下 :"%factory)
        print (list(conf.sections()))
    return periods

def GetPointTable(factory = "吉林大成"):
    '''
     获取数据集:
     factory : 项目名称 / {"吉林大成","宁夏伊品2#","宁夏伊品3#"}
     return ; 项目点表
    '''
    conf = _getconf()
    details = dict(conf.items("details"))
    if factory not in details:
        print ("factory 输入有误,当前配置项目如下:")
        print (list(details.keys()))
        return []
    des = pd.read_csv(os.path.join(INIT,details[factory]))
    return des

def DrawPeriods(df,periods,labels):
    plt.figure(figsize=(18,6))
    plt.plot(df)
    ymax = df.max()
    periods = pd.DatetimeIndex(periods)
    for i,xindex in enumerate(periods):
        plt.vlines(xindex, 0, ymax, colors = "red", linestyles = "dashed",linewidth=3.5)
        if i!=0:
            xtext = (periods[i]-periods[i-1])*(1/6)+periods[i-1]
            coff = 0.95 if i%2==0 else  0.75
            ytext = ymax*coff
            plt.text(xtext, ytext, labels[i-1],fontdict={'family' : 'SimHei','weight' : 'normal','size': 9})
    plt.show()

@costtime
def GeneratePeriods(data,column='mean',varthreshold = 1900,threshold = 8640.0,show=True):
    df = data[column].sort_index() ## df is a Series
    s = df[df>varthreshold]
    s1 =pd.Series(s.index,index = s.index).diff().dt.total_seconds()
    s2 =pd.Series(s.index,index = s.index).diff(-1).dt.total_seconds().abs()
    periods = s1[s1>threshold].append(s2[s2>threshold])\
        .append(s.iloc[[0,-1]])
    bz = periods.index[0]==df.index[0]
    periods=periods.append(df.iloc[[0,-1]]).sort_index().index.drop_duplicates()
    if bz:
        p1,p2= 'Discharge','Working'
    else:
        p1,p2= 'Working','Discharge'
    N = math.floor(len(periods)/2)
    labels = np.array([[p1+str(i+1),p2+str(i+1)] for i in range(N)]).flatten().tolist()[:len(periods)-1]
    _periods= periods.strftime("%Y-%m-%d %H:%M:%S").tolist()
    result ={"periods":_periods,"labels":labels}
    if show:
        DrawPeriods(df,periods,labels)
    return result

@costtime
def GetProcessData(factory='吉林大成',month = '2018_09',keyword = ['加氨槽'] ,\
             column = 'mean',resample = True, freq = '600S',period =True):
    '''
     获取数据集:
     factory : 项目名称 / {"吉林大成","宁夏伊品2#","宁夏伊品3#"}
     month : 月份名称/ 子目录  /  {"2018_09","2018_10","2018_11","2018_12","2019_01"}
     keyword : 测点关键字 (组别)  {"加氨槽","净烟气","原烟气","浓缩段","浓缩槽","吸收段"..}
         类型为str 时 : 单选该组别所有变量
         类型为list 时 : 多选组别所有变量
         类型为dict 时 : 各组自定义筛选变量列表,值为空默认全选
     column : 原始数据提取的字段名 / {'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'mad',
       'skew', 'kurt', 'rms', 'bad'}
     resample : 是否降采样
     freq : 降采样周期
     period : 是否根据周期进行Period赋值
     
     return ; 合并后的数据集
     '''
    des = GetPointTable(factory)
    if len(des)==0:
        return
    dirname = os.path.join(os.path.join(RAWDATA,factory),month)
    if not os.path.exists(dirname):
        print ("未发现处理数据")
        return pd.DataFrame()
    
    keywords = []
    if isinstance(keyword,str):
        keywords = des[des['关键字']==keyword]['名称'].tolist()
    if isinstance(keyword,list):
        keywords = des[des['关键字'].isin(keyword)]['名称'].tolist()
    if isinstance(keyword,dict):
        for key in keyword:
            if len(keyword[key])!=0:
                keywords.extend(keyword[key])
            else:
                keywords.extend(des[des['关键字']==key]['名称'].tolist())
       
#     des1 = des[(des['名称'].isin(keywords))&(des['名称'].isin(map(lambda x:x.split(".")[0],os.listdir(dirname))))][['名称','描述']]
    des1 = des[(des['名称'].isin(keywords))&(des['名称'].isin(map(lambda x:os.path.splitext(x)[0],os.listdir(dirname))))]\
    [['名称','描述']]
    dfs = [ReadDfFile(os.path.join(dirname,i))[[column]] for i in (des1['名称']+".pk")]
    if len(dfs) ==0 :
        print ("未发现测点数据")
        return pd.DataFrame()
    if resample:
        res = ResampleData(dfs,column =column, names = des1['描述'],freq=freq)
    else:
        res = pd.concat(dfs,axis=1)
        res.columns = des1['描述']
    if period:
        periods = GetPeriods(factory,month)
        if len(periods)!=0: 
            res['Period'] = pd.cut(res.index,pd.Series(\
            pd.to_datetime(periods['periods'])),labels=periods['labels'])
    return res

@costtime
def ResampleData(dfs,column = 'mean',names = [],freq = "600S",Period = {}):
    '''
     关联降采样函数
     dfs: 待降采样的 df 组成的list ,要求每个df的index 都是 DateTimeIndex类型
     column : 各df 中需要选取的 合并的字段名称， 默认为 mean
     names ;  各df合并后 的数据集的 columns， 默认为空，则names = [col1,col2...]
     freq : 降采样周期, 默认为 600S , 即10min
     Period :  项目周期配置  , 通过 GetPeriods(factory,month) 获取的dict
     return : 合并后降采样数据集
     '''
    df = pd.concat( map(lambda x:x[column],dfs),axis=1)
    if len(names)==0:
        names = ["col%d"%(i+1) for i in range(len(dfs))]
    df.columns = names
    df1 = df.resample(freq).mean()
    return df1

@costtime
def Scalerdf(df):
    '''
     归一化函数:
     df : 待归一化DataFrame
     return : df中所有数值型列的归一化 DataFrame
     '''
    df1 = df.select_dtypes(include=['int64','float64'])
    scaler = _MinMaxScaler()
    df_new = pd.DataFrame(scaler.fit_transform(df1),\
                          index = df1.index,\
                          columns= df1.columns)
    return df_new

def SaveDfFile(df,path,*args,**args1):
    filetype = os.path.splitext(path)[-1]
    if filetype=='.zip':
        if "compression" not in args1:
            args1['compression'] ='zip'
    if filetype in ['.pk']:
        df.to_pickle(path,*args,**args1)
    if filetype in ['.zip','.csv','.txt']:
        df.to_csv(path,*args,**args1)
    if filetype in ['.xlsx','.xls']:
        df.to_excel(path,*args,**args1)

def ReadDfFile(path,*args,**args1):
    '''
     通用pandas读取文件函数:
     path : 待读取文件的路径, 支持根据文件后缀格式 自动选取对应 pd读取函数
     return : 文件读取的 DataFrame
     '''
    filetype = os.path.splitext(path)[-1]
    d = {".csv":pd.read_csv,".xls":pd.read_excel,".xlsx":pd.read_excel,\
         ".txt":pd.read_csv,".pk":pd.read_pickle}
    if filetype not in list(d.keys()):
        print ("文件格式不支持")
        return 
    if filetype not in ['.pk','.xls','.xlsx']:
        args1['engine'] ='python'
    return d[filetype](path,*args,**args1)