#!/usr/bin/env python

import pandas as pd
import multiprocessing
import logging
import os,datetime
import time
from zipfile import ZipFile
import pandas as pd
import numpy as np
from numpy import *
import argparse
import math 
from khcore import ReadDfFile,SaveDfFile

LOGGING = './log'

def InitLog(workname):
    ## log配置   
    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.DEBUG)
#     now = datetime.datetime.now().strftime("%Y%m%d %H%M%S")
    if not os.path.exists(LOGGING):
        os.makedirs(LOGGING)
    handler = logging.FileHandler(os.path.join(LOGGING,"%s_log.txt"%workname))
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger 

def rms(Z):
    return sqrt(mean(square(Z)))

def Statistics(df,col = 'value'):
    if len(df)!=0:
        s = df[col].describe()
        s1 = df[col].agg(['mad','skew','kurt'])
        s2 = pd.concat([s,s1])
        s2['rms'] = rms(df[col])
        s2['bad'] = len(df[df['quality']!='QUALITY_GOOD(0)'])
        return s2

def GetData(filepath,chunksize=None,engine = "python",**args):
    myzip=ZipFile(filepath)
    f=myzip.open(myzip.filelist[0])
    if chunksize!=None:
        df = pd.read_csv(f,encoding='gb2312',chunksize=chunksize,**args)
    else:
        df = pd.read_csv(f,encoding='gb2312',**args)
        f.close()
    return df,f

def Stander(df):
    df = df.dropna()
    df = df[df['时间'].str.match('[0-9]')]
    df['时间'] = pd.to_datetime(df['时间'])
    df = df.rename(columns = {"时间":"time","数值":"value","质量":"quality"})
    return df
        
def GetDATA(filepath,savepath,freq = '10S',chunksize = 50000,logger=None,workname='worker',cpu = 1,total = 1,**args1):
    try:
        logger = InitLog(workname)
        n = int(workname.split("_")[-1])-1
        wn = (n//cpu+1) / math.ceil(total/cpu)  ## 当前pool 中进程处理进度系数
        size = os.path.getsize(filepath)
        coefficient = 0.37  ## 文件折算值 = 文件行数/ 文件size 
        logger.info("%s : %s filesize: %0.3fM"%(workname,filepath,size/1024**2))
        dfs,f = GetData(filepath,chunksize =chunksize)
        res = []
        rawdf = pd.DataFrame()
        totalcost = None
        N = 1.0
        for df in dfs:
            st = time.time()
            df = Stander(df)
            df = rawdf.append(df)
            bins = pd.date_range(df.time.min(),df.time.max(),freq = freq)
            rawdf = df[df['time']>=bins[-1]] ## 非整周期数据 轮入下轮计算
            temp = df[df['time']<bins[-1]].resample(freq,on = 'time').apply(Statistics)
            res.append(temp)
            et = time.time()
            if totalcost==None:
                totalcost = (size*coefficient/chunksize)*(et-st)/wn*1.1 ## 估算文件整体耗时
#             logger.info("%s/batch timesmap : %s"%(workname,temp.index[0]))  
            logger.info("%s/batch  %s : %s"%(workname,filepath,temp.shape))
            logger.info("%s/batch.cost : %0.2fs "%(workname,et-st))
            progress = min([chunksize*N/(size*coefficient),1.0])*wn
            logger.info("%s/batch.progress : %0.2f%%  Remaining time: %0.2fs "%(workname,progress*100,totalcost*(1-progress)))
            N+=1
    
        res = pd.concat(res)
        if os.path.dirname(savepath)!='' and not os.path.exists(os.path.dirname(savepath)):
            os.makedirs(os.path.dirname(savepath))
        try:
            args1.pop("savetype")
        except:
            pass
        SaveDfFile(res,savepath,**args1)
        f.close()
        logger.info("%s/success %s save "%(workname,savepath))
        return "success :  %s"%(savepath)
    except Exception as e:
        logger.error("%s/error %s : %s"%(workname,filepath,e))
        return "error : <file> %s , %s"%(filepath,e)

def worker(args):
    return GetDATA(**args)
        
def GetPool(cpu_count=-1):
    if cpu_count==-1:
        cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpu_count)
    return pool,cpu_count

def ParallelCal(cpu_count=-1,files = [], savepaths = [], freq = '10S',\
                chunksize = 50000, savetype = 'pk',udf = None,\
                callback = None ,**args1):
    '''
     原始db文件并行处理函数:
     cpu_count : 并行计算进程数 默认为-1, 即全部调用系统全部CPU
     files ; 待处理文件的路径列表
     savepaths : 对应处理文件处理后保存的文件路径列表 , 默认为空, 则在当前process文件夹中保存位置对称文件
     freq : 降采样周期 ,默认为10s
     chunksize : 文件切片行数 , 默认为 50000
     savetype : 保存文件格式 / {'pk', 'pickle', 'csv', 'zip', 'xls', 'xlsx', 'excel'}
     udf :  用户自定义处理逻辑函数
     
     return : 处理成功的文件路径列表
     '''
    st = time.time()
    savetype = savetype.replace('.','')
    savetypes = {"pk":".pk", "pickle":".pk",'csv':".csv",'txt':".txt",\
     'zip':".zip",'xls':".xls",'xlsx':".xlsx",'excel':".xlsx"}
    if savetype not in savetypes:
        print ("文件格式不支持")
        return 
    if len(savepaths)==0:
        savepaths = list(map(lambda x:x.replace("db","process").\
           replace(os.path.splitext(x)[-1],savetypes[savetype]),files))
    else:
        if len(files) != len(savepaths):
            print ("存储文件和输入文件个数不一致")
            return
        
    pool,cpu_count =  GetPool(cpu_count)
    savepaths = list(map(lambda x:os.path.splitext(x)[0]+savetypes[savetype],savepaths))   
    d = {"freq":freq,"chunksize":chunksize,"savetype":savetype,\
         "cpu":cpu_count,"total":len(files)}
    d.update(args1)
    ppargs = []
    N = 1
    for filepath,savepath in zip(files,savepaths):
        tempdict = {"filepath":filepath,"savepath":savepath}
        tempdict.update(d)
        tempdict['workname'] = "worker_%d"%N
        N+=1
        ppargs.append(tempdict)
        
    logger = InitLog("MAIN")
    logger.info("freq : %s"%(freq))
    logger.info("chunksize : %s"%(chunksize))
    logger.info("savetype : %s"%(savetype))
    logger.info("get args : %d"%(len(ppargs)))
    logger.info("cpu_count : %d"%cpu_count)
    logger.info("start processing .....")
    if udf == None:
        func = worker
    else:
        func = udf
    res_list = []
    
    for i in ppargs:
        res = pool.apply_async(func=func, args=(i,), callback=callback)
        res_list.append(res)
    
    pool.close()
    logger.info("pool is closed")
    pool.join()
    result = [res.get() for res in res_list] 
    pool.terminate()
    et = time.time()
    logger.info("end processing ")
    logger.info("process cost : %0.2fs "%(et-st))
    return result

