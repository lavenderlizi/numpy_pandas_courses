# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 20:39:39 2020

@author: lili6
"""

import numpy as np

'''数组创建'''
a=np.array([[1,2,3],[5,6,7]])
a.dtype#类型
a.shape#对象元素的个数，n行m列
a.size#n*m
a=np.array([[1,2,3],[5,6,7],[7,8,9]])
a.ndim#维度的数量,中括号的数量,len(a.shape)

##通过输入list
b=np.array([[1.2,2.4],[0.3,4.1]])
#b=np.array([1.2,2.4],[0.3,4.1])是错误的！

###通过输入元组效果一样，其中可以使用字符或者其他的元素
c=np.array((('a',1,2),(3,4,5)))

###通过内置
np.zeros((6,5))
np.ones((2,3))
np.arange(5)###注意默认从0开始
np.arange(3,15,2)####可以实践并思考三个数的含义


#####改变shape
np.arange(0,12).reshape(3,4)
a=np.arange(0,12)
a.shape=(3,4)#####改变形状的另一种方式

np.linspace(0, 1, 1000)#画图常用
np.linspace(0,6,3)

np.random.random((3,4))
#####尝试下增加一维度，理解它的生成方式
np.random.random((3,4,2))
np.random.random((3,4,2)).ndim
len(np.random.random((3,4,2)).shape)
a=np.random.randn(3)##正态分布
np.random.randn(3,1)##最好写成这样
np.random.standard_normal(5)
np.random.standard_normal((5,1))




'''运算与broadcast广播机制'''

####加减乘除的机制
b+1
b**2###注意不是用^ !


d=np.ones((2,3))/5
e=np.ones((1,3))-3
d+e
f=np.arange(2).reshape(2,1)
d*f
g=np.arange(2).reshape(1,2)
d*g####会提示报错operands could not be broadcast together with shapes (2,3) (2,) 

f=np.arange(3).reshape(3,1)
np.dot(d,f)####矩阵乘法,注意顺序
np.matmul(d,f)


#####其他运算
np.sin(b)
np.sqrt(b)
np.log(b)
np.exp(b)

a+=1####但是这种自增运算并没有建立副本，要注意不要多运行几次之后把数据改变了

a*=2

###转置
k=d.T
k=d.transpose()

#####聚合函数
b.sum()
b.min()
b.min(axis=0)#找出每列最小
b.min(axis=1)#找出每行最小
b.max()
b.std()

#####为了减少循环，可以使用
b.mean(axis=1)
np.apply_along_axis(np.mean,axis=1,arr=b)###对b数组的每行作

def example(x):
    y=x**2
    y=y.mean()
    return y
np.apply_along_axis(example,axis=1,arr=b)#####对每行做平方后求均值

'''定义维数的重要性'''
w=np.random.randn(5)#rank1 array
print(w.shape)
print(w)
print(w.T)
print(np.dot(w,w.T))#出现的是一个数，不是想要的结果

w=np.random.randn(5,1)
print(w.shape)
print(w)
print(w.T)
print(np.dot(w,w.T))

'''小练习均方误差MSE'''


'''索引和切片'''
b=np.array([[1.2,2.4],[0.3,4.1]])
b[2]###发现出错，注意从0开始
b[1]
b[1,1]#####如果想要特定值
##切片
h=np.arange(9,16).reshape(7,1)
h[1:5:2]####取第2到6个元素，间隔为2表示每隔一个元素抽取一个
h[-2]
i=np.arange(11,20).reshape((3,3))
i[2,:]
i[[0,2],0:2]



######元素选取，利用判断句，布尔数组
j=np.random.random((5,5))
j<0.5
j[j<0.5]
j[j<0.5]=1####可以以此改变这些数
j




'''数组操作'''
l=np.ones((3,3))
m=np.zeros((3,3))
np.vstack((l,m))###垂直叠放
np.hstack((l,m))###水平叠放

o=np.array([0,1,2])
p=np.array([3,4,5])
q=np.array([6,7,8])
np.column_stack((o,p,q))
np.row_stack((o,p,q))
np.column_stack((o,l,p,q))
np.row_stack((o,l,p,q))


o,p,q=np.hsplit(l,3)
k=np.arange(16).reshape((4,4))
r,s=np.hsplit(k,2)###水平切分
r,s=np.vsplit(k,2)###垂直切分
r,s,t=np.split(k,[1,3],axis=1)###按列切
r,s,t=np.split(k,[1,3],axis=0)###按行切


####创建副本
m=l
l[1,1]=6
l
m
m=l.copy()
l[1,1]=5
l
m


#######画函数图
x = np.linspace(0, 1, 1000)
y = x/(1+x)

import matplotlib.pyplot as plt 
plt.plot(x,y,'r',linewidth=2)



