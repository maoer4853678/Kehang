#!/usr/bin/env python
# coding: utf-8

import Khcore1 as kc1
import MyEcharts as me
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn import linear_model

def GetFreq(df):
    t = df.index.freq
    if len(t.kwds):
        S=list(t.kwds.values())[0]*7*24*3600
    else:
        S = t.delta.total_seconds()
    N = t.n
    return N,S

def SensorCal(df, sensor = "ATTLT_3",window1 = 3,window2 = 10,overwrite=True, name = None):
    '''
    出口SO2 传感器校核函数
    df : 待校准的数据源,应包含 出口SO2 和 出口O2
    sensor : 校准参照测点, 一般参照 出口O2
    window1 : 过程滑动窗口长度 单位为 min , 默认为 3min
    window2 : 合并窗口长度 单位为 min, 即若判别两个区间距离小于 window2, 则合并为一个窗口
    overwrite : 是否覆盖原测点变量, 默认覆盖
    name :  存储查看修正后 效果html, 默认不存储
    '''
    
    N,S = GetFreq(df)
    if isinstance(window1,str):
        window1 = pd.to_timedelta(window1).total_seconds()
    else:
        window1*=S
    if isinstance(window2,str):
        window2 = pd.to_timedelta(window2).total_seconds()
    else:
        window2*=S
    
    ## 传感器失效区间 判别
    data = df.sort_index()
    temp = data[sensor].diff()
    temp1 = temp.abs().rolling(window = "%ds"%window1).max()
    temp2 = temp1[temp1>0.3].to_frame()
    temp2['diff']= ((pd.Series(temp2.index).diff().dt.total_seconds()).values/S)
    temp2['diff']= temp2['diff'].fillna(100)
    ## 生成传感器失效区间 
    res = pd.DataFrame()
    res['start']= temp2[temp2['diff']>(N+0.1)].index
    temp3 = temp2.index.get_indexer(temp2[temp2['diff']>(N+0.1)].iloc[1:].index)-1
    res['end']=temp2.iloc[temp3].index.tolist()+[temp2.index[-1]]
    res['start'] = res['start']- pd.to_timedelta(S,unit ='s')
    res['end'] = res['end']- pd.to_timedelta(S,unit ='s')
    res["temp"] = [np.nan]+((res.start.iloc[1:]-res.end.iloc[:-1].values).dt.total_seconds()).tolist()
    res["temp"] = res["temp"].fillna(100)
    bins = res[res['temp']>window2].index.tolist()
    if len(bins):
        if bins[0]!=res.index[0]:
            bins= [res.index[0]]+bins
        if bins[-1]!=res.index[-1]:
            bins+=[res.index[-1]+1]
        res['temp1'] = pd.cut(res.index,bins,right=False)
        res = res.groupby("temp1").agg({"start":"min","end":"max"})
    res['time'] = res['end']-res['start']
    res.index = range(len(res))
    
    ## 根据传感器失效区间 对O2 和 SO2 插值
    if not overwrite:
        raw = data.copy()
    for i in res.index:
        t = data[(data.index>=res.loc[i,'start'])&(data.index<=res.loc[i,'end'])]
        data.loc[t.index,:] = np.nan
    data = data.resample("%ds"%S).interpolate()
    if not overwrite:
        data.columns = data.columns+"_bd"
        for col in raw.columns:
            data[col] = raw[col]
    if name:
        me.Plot_LineBar(data, name = name, overwrite=True)
        print ("save image as %s"%name)
    return data


def AutoLabel(df,column='',th =95 ,center_window = None, left_window = None, right_window = None,                      merge_window = None,  varname='label',filename = None):
    '''
    测点自动打标签函数
    df : 待打标签的数据集
    column :  标签参考变量名称
    th  :  变量超限阈值
    center_window  : int/format  超限阈值后左右窗口长度, 优先级高
    left_window :  int/format 左侧窗口向左偏移量, 默认为 None , 优先级低
    right_window : int/format  右侧窗口向右偏移量, 默认为 None , 优先级低
    merge_window : int/format 合并窗口长度 , 即若判别两个区间距离小于合并窗口长度, 则合并为一个窗口  
    varname : 标签字段名称 ,默认 label
    filename :  存储查看修正后 效果html, 默认不存储
    '''  
    data2 = df.sort_index()
    N,S = GetFreq(data2)
    if center_window:
        if isinstance(center_window,str):
            center_window = pd.to_timedelta(center_window).total_seconds()/S
        left = (center_window)*N-1
        right = (center_window-1)*N+1
    else:
        if isinstance(left_window,str):
            left_window = pd.to_timedelta(left_window).total_seconds()/S
        if isinstance(right_window,str):
            right_window = pd.to_timedelta(right_window).total_seconds()/S
        left = (left_window)*N-1
        right = (right_window-1)*N+1
    if merge_window:
        merge_window = pd.to_timedelta(merge_window).total_seconds()
    left*=S
    right*=S

    t = data2[data2[column]>th]
    t['diff']= ((pd.Series(t.index).diff().dt.total_seconds()).values/S)
    t['diff']= t['diff'].fillna(100)
    ## 生成 正样本标签 区间 
    res1 = pd.DataFrame()
    res1['start']= t[t['diff']>(N+0.1)].index
    t2 = t.index.get_indexer(t[t['diff']>(N+0.1)].iloc[1:].index)-1
    res1['end']=t.iloc[t2].index.tolist()+[t.index[-1]]
    res1['start'] = res1['start']- pd.to_timedelta(left,unit ='s')
    res1['end'] = res1['end']+ pd.to_timedelta(right,unit ='s')
    res1["temp"] = [np.nan]+((res1.start.iloc[1:]-res1.end.iloc[:-1].values).dt.total_seconds()).tolist()
    res1["temp"] = res1["temp"].fillna(100)
    bins = res1[res1['temp']>merge_window].index.tolist()
    if len(bins):
        if bins[0]!=res1.index[0]:
            bins= [res1.index[0]]+bins
        if bins[-1]!=res1.index[-1]:
            bins+=[res1.index[-1]+1]
        res1['temp1'] = pd.cut(res1.index,bins,right=False)
        res2 = res1.groupby("temp1").agg({"start":"min","end":"max"})
    else:
        res2 = res1
    res2['time'] = res2['end']-res2['start']
    res2.index = range(len(res2))
    
    for i in res2.index:
        t3 = data2[(data2.index>=res2.loc[i,'start'])&(data2.index<=res2.loc[i,'end'])]
        data2.loc[t3.index,varname] = N
    data2[varname] = data2[varname].fillna(0)
    if filename:
        temp = (data2-data2.min())/(data2.max()-data2.min())
        me.Plot_LineBar(temp, name = filename, overwrite=True)
        print ("save image as %s"%filename)
    return data2


class SingleTsFeature: 
    def __init__ (self,df):
        '''
        df : Series/DataFrame
        '''
        if isinstance(df,type(pd.Series())):
            self._df = df.to_frame()
        else:
            self._df = df.copy()
        if str(self._df.index.dtype)!="datetime64[ns]":
            raise Exception("数据索引不为时间格式")
        self._columns = self._df.columns
        if len(self._columns)==1:
            self._column = self._columns[0]
        else:
            self._column = None
        self._scale = None
    
    def _checkcolumn(self,column):
        if column ==None:
            if self._column ==None:
                raise Exception("原始数据字段不唯一, 请指定字段名")
            else:
                column = self._column
        return column
        
    def __str__(self):
        return str(self.df)
    
    def add_label(self,column= None ,th =100 ,center_window = None, left_window = None, right_window = None,                      merge_window = None ,varname = "label" ,filename = None):
        '''
        自动标签
        column :  标签参考变量名称
        th  :  变量超限阈值
        center_window  : int/format  超限阈值后左右窗口长度, 优先级高
        left_window :  int/format 左侧窗口向左偏移量, 默认为 None , 优先级低
        right_window : int/format  右侧窗口向右偏移量, 默认为 None , 优先级低
        merge_window : int/format 合并窗口长度 , 即若判别两个区间距离小于合并窗口长度, 则合并为一个窗口  
        varname : 标签字段名称 ,默认 label
        filename :  存储查看修正后 效果html, 默认不存储
        '''  
        column = self._checkcolumn(column)
        self._df = AutoLabel(self._df,column=column,th =th ,center_window = center_window, left_window = left_window,                  right_window = right_window,  merge_window = merge_window,varname=varname,  filename = filename)
        print ("add label successful ")
    
    def add_slope(self,column= None ,window = "10min", varname = "slope"):
        '''
        加工斜率特征
        column :  参考变量名称
        window  : int/format  滑动窗口长度
        varname :  特征名称, 默认为 slope
        '''
        column = self._checkcolumn(column)
        if not isinstance(window,str):
            N,S = GetFreq(df)
            window = "%dS"%(int(window*S))
            print ("slope window : %s"%window)
        self._df[varname] = self._df[self._column].rolling(window=window).apply(kc1.gradient)
        print ("add_slope successful ")
        
    def add_roll_quantile(self,column= None ,q = 0.75, window = "1min", varname = ""):
        '''
        加工滑动分位数特征
        column :  参考变量名称
        q : 分位数
        window  : int/format  滑动窗口长度
        varname :  特征名称
        '''
        if not len(varname):
            raise Exception("特征变量名称不能为空")
        column = self._checkcolumn(column)
        if not isinstance(window,str):
            N,S = GetFreq(df)
            window = "%dS"%(int(window*S))
            print ("%s window : %s"%(varname,window))
        self._df[varname] = self._df[self._column].rolling(window=window).quantile(q)
        print ("add %s successful "%varname)
    
    def add_roll_func(self,column= None ,func = "mean", window = "1min", varname = ""):
        '''
        加工滑动特征
        column :  参考变量名称
        func : 滑动函数 {'mean','max','min','std','median',自定义函数}
        window  : int/format  滑动窗口长度
        varname :  特征名称
        '''
        if not len(varname):
            raise Exception("特征变量名称不能为空")
        column = self._checkcolumn(column)
        if not isinstance(window,str):
            N,S = GetFreq(df)
            window = "%dS"%(int(window*S))
            print ("%s window : %s"%(varname,window))
        self._df[varname] = self._df[self._column].rolling(window=window).agg(func)

    def scaler(self,stype = "mm",columns = [],exclude = []):
        '''
        归一化/标准化
        stype: {"mm":MinMaxScaler,"stand":StandardScaler}
        columns : 待标准化的字段列表, 默认为空, 则为目前数据的全部字段列
        exclude : 不需标准化的字段列表
        '''
        stypes = {"mm":MinMaxScaler,"stand":StandardScaler}
        scale = stypes[stype]()
        if not len(columns):
            columns = self._df.columns.tolist()
        if not len(exclude):
            columns = list(set(columns)-set(exclude))
        self._scale = scale.fit(self._df[columns])
        self._sdf = pd.DataFrame(scale.transform(self._df[columns]),columns = columns,                        index = self._df.index)
        return self._sdf
    
    def GetScale(self):
        '''
        获取标准化实例
        '''
        return self._scale
    
    def GetDf(self):
        '''
        获取特征数据
        '''
        return self._df


if __name__ =="__main__":
    ## 温度数据
    names = ["TE_1003a","TE_1003b","TE_1003c"]
    datas=  []
    for name in names:
        datas.append(kc1.ReadRawData(name))
    data = pd.concat(datas,axis=1)
    data= data[data.index>"2019-06-13 19:00:00"]   ## 2019-06-13 19:00:00 之前属于数据测点调试阶段
    data.columns = names
    data = data.resample("1min").mean()
    data1 = data.max(axis=1)
    
    ## 利用SingleTsFeature 根据温度最大值 生成label
    s = SingleTsFeature(data1)
    s.add_label(th=62.4,left_window = '60min',right_window="5min",merge_window="20min")
    
    ## 平均温度 为特殊加工变量， 需要用户生成
    ## 利用SingleTsFeature 根据平均温度 生成特征
    df = kc1.GenerateAvgTemp(["TE_1003a","TE_1003b","TE_1003c"],column='value', factory='中碳能源', month='2019_06')
    df= df[df.index>"2019-06-13 19:00:00"]
    df = df.resample("1min").mean()
    df.columns = ['TE_1003']
    
    s1 = SingleTsFeature(df)
    s1.add_roll_quantile(q = 0.75,window='1H',varname = "avg")
    s1.add_slope(window="4H")
    s1.add_roll_func(func = 'mean',window='1h',varname = "mean")
    
    ## diff 为特殊加工变量， 需要用户生成
    df1 = pd.concat([data[['TE_1003a','TE_1003b','TE_1003c']],s1.GetDf(),s.GetDf()[['label']]],axis=1)
    df1['diff'] = df1[['TE_1003a','TE_1003b','TE_1003c']].max(axis=1)-df1['TE_1003']  ## 温度峰峰值
    df1['diff'] = df1['diff'].rolling("1H").mean()

    scaler = MinMaxScaler()
    ## 采用 Logistic Regression 模型
    X= df1[['diff','avg','slope']].values
    X1=scaler.fit_transform(X)
    Y = df1[['label']].values
    
    model = linear_model.LogisticRegression( solver = 'saga', class_weight = 'balanced', max_iter=1000)
    model.fit(X1,Y)
    print(model.score(X1,Y))
    print (model.coef_)
    df1['TE_1003a']/=60
    df1['Logistic'] = model.predict_proba(X1)[:,1]
    me.Plot_LineBar(df1,columns=['label','Logistic'],name="label_Logistic",overwrite=True)



