# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 20:49:56 2020

@author: lili6
"""

import pandas as pd
import numpy as np

'''Series'''
##创建series
a=pd.Series([10,44,-9,8])
a=pd.Series([10,44,-9,8],index=['a','b','c','d'])
a.values
a.index

b=np.array([1,2,3,4,5])
b=pd.Series(b)
b=np.array([1,2,3,4,5]).reshape(5,1)
b=pd.Series(b)#报错
b=pd.Series(b[:,0])
b=np.array([1,2,3,4,5]).reshape(5,)
b=pd.Series(b)

##字典创建series对象
mydict={'a':233,'b':2333,'c':23333,'d':233333}
d=pd.Series(mydict)

mydict={}
mydict['a']=233
mydict['b']=2333
mydict['c']=23333
mydict[1]=6
for i in range(6):
    mydict[i]=i*2
d=pd.Series(mydict)
d=pd.Series(mydict,index=['a','b','c','d'])

####切片
a[2]
a['b']
a[0:3]
#Q：选取倒数几个数？
a[[0,2]]
a[['b','c']]
a[2]=5
a
a[['b','c']]=[3,7]
a


###选取
a>8
a[a>8]
a.isin([10,2])
a[a.isin([10,2])]


###运算
a/2
np.log(a)
np.sin(a)
np.exp(a)

###去重
c=pd.Series([1,2,2,3,0,1,0,4],index=['q','w','e','r','t','y','u','i'])
c.unique()
set(c)
c.value_counts()

c.drop_duplicates()

####关于nan
c=pd.Series([1,np.NaN,2,3,0,1,0,4],index=['q','w','e','r','t','y','u','i'])
c[c.isna()]
c.isnull()
c.notnull()
c.dropna()
c=pd.Series([1,np.NaN,2,3,0,1,0,4],index=['q','w','e','r','t','y','u','i'])
c.fillna(666)

'''DataFrame'''
###字典创建
df=pd.DataFrame({'年龄':[23,25,26],'性别':[1,0,0]},
                index=['student1','student2','student2'])
###数组创建
df=pd.DataFrame(np.arange(25).reshape(5,5),
                index=['red','blue','yellow','white','black'],
                columns=['ball','pen','pencil','paper','cup'])

'''小练习生成DataFrame'''

df.columns
df.values
df.index
df.index.tolist()

##添加列
df[1]=[1,2,3,4,5]
df['a']=['q','w','er',123,234]
df['b']=12
#可以类比之前的字典法，利用df=pd.DataFrame()

##添加行
df.append([[1,2,3,5,6,7,8]])
df1=df.append({'第0个特征':555,'a':66,'b':123},ignore_index=True)

####切片
##取列
df['a']
df.a

##取行
df.loc['red']###取行,根据index
df.loc[2]#报错

df.iloc[1]###i指integer，表示给行号
df.iloc[3,2]
df.iloc[1,:]
df



'''通过实例学习统计函数'''
######从文件读取和存储到文件
'E:\\。。。\\厦大\\wiser课程部教学\\2019\\Numpy与Pandas\\train.csv'
trainpath='E:\\everything about machine learning\\kaggle\\starting项目的数据集\\Kaggle-master\\Datasets\\Titanic\\train.csv'
testpath='E:\\everything about machine learning\\kaggle\\starting项目的数据集\\Kaggle-master\\Datasets\\Titanic\\test.csv'
train=pd.read_csv(trainpath)
train=pd.read_csv(trainpath,index_col='PassengerId')
test=pd.read_csv(testpath,index_col='PassengerId')
train.to_csv('')###加入path


train.index
train.columns

'''小练习:选取年龄小于20岁人的性别并统计数量'''

#统计函数
train.info()
train.describe()
type(train.describe())
train.describe().loc['count']
train.describe().loc['max']

#统计函数
train.sort_values('Age',ascending=False)#根据值排序

train['Age'].cumsum()#累加
train['Age'].cumprod()#累乘

train['Age'].rolling(2).sum()#纵向（0轴方向）上以两个元素为单位做运算
train['Age'].rolling(3).mean()

train['Fare'].diff(1)

#相关性
corr=train.corr()
corr
import seaborn as sns 
sns.heatmap(corr,annot=True,vmax=1,vmin=0,xticklabels=True,yticklabels=True,square=True,cmap="YlGnBu")

Survived_0=train.Pclass[train.Survived==0].value_counts()
Survived_1=train.Pclass[train.Survived==1].value_counts()

group=train.groupby(['SibSp','Survived'])
group.sum()
group.mean()
group.apply(lambda x:x.max())

#####特征处理
train['age_deal']=[x//5 for x in train['Age']]

####哑变量，特征因子化
dummies_Cabin=pd.get_dummies(train['Cabin'],prefix='Cabin')


####拼接操作
train.join(dummies_Cabin)

dummies_Cabin['Id']=train.index
train['Id']=train.index
pd.merge(train,dummies_Cabin,on='Id',how="right")

df=pd.concat([train,dummies_Cabin],axis=1)###



####删除列
df.drop(['Pclass','Name','Sex','Ticket','Cabin','Embarked'],axis=1,inplace=True)
df.columns
del df['Cabin_A16']


####机器学习scikit-learn库
from sklearn import linear_model
train_df=pd.read_csv('E:\\everything about machine learning\\kaggle\\starting项目的数据集\\Kaggle-master\\Datasets\\Titanic\\after.csv')
train_df=pd.read_csv('E:\\everything about machine learning\\kaggle\\starting项目的数据集\\Kaggle-master\\Datasets\\Titanic\\after.csv',header=0)
train_np = train_df.values####是个array
y=train_np[:,0]
x=train_np[:,1:]
model=linear_model.LogisticRegression(penalty='l2',tol=1e-6)
model.fit(x,y)
model
model.intercept_
model.coef_
#model.predict(test)
