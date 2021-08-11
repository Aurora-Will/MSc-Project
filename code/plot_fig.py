
# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

def get_x_y(path):
  df=pd.read_csv(path,sep=',',usecols=[1,2])
  x = []
  y = []
  for index,row in df.iterrows():
    x.append(row['Step'])
    y.append(row['Value'])
  return x,y
train_acc_x,train_acc_y= get_x_y('a/run-.-tag-Train_Acc.csv')

train_loss_x,train_loss_y = get_x_y('a/run-.-tag-Train_Loss.csv')

val_acc_x,val_acc_y = get_x_y('a/run-.-tag-Validation_Acc.csv')
val_loss_x,val_loss_y = get_x_y('a/run-.-tag-Validation_Loss.csv')

  
#这里导入你自己的数据
#......
#......
#x_axix，train_pn_dis这些都是长度相同的list()
  
#开始画图
# sub_axix = filter(lambda x:x%200 == 0, x_axix)
plt.title('Train Model Analysis')
# plt.plot(train_acc_x, train_acc_y, color='green', label='train accuracy')
# plt.plot(val_acc_x, val_acc_y, color='red', label='val accuracy')


plt.plot(train_loss_x, train_loss_y, color='skyblue', label='train loss')
plt.plot(val_loss_x, val_loss_y, color='blue', label='val loss')
plt.legend() # 显示图例
  
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()
#python 一个折线图绘制多个曲线