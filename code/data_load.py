import torch

import pandas as pd

import numpy as np
import time


def get_time_list(dd):
    timeArray = time.localtime(dd/1000)
    otherStyleTime = time.strftime("%Y-%m-%d-%H-%M-%S", timeArray)
    ddd = [ int(x) for x in otherStyleTime.split('-')[1:]]

    return ddd

# ['Unnamed: 0', 'cache_memory', 'collection_id', 'cpu_usage', 'instance_index', 'memory_usage', 'priority', 'resubmit_time', 'scheduling_class', 'time', 'type']
###  0 time,instance_index,collection_id,type,priority,machine_id,start_time,end_time,
###  8 cpus,memory,

### 10  0_time,0_type,0_collection_id,0_scheduling_class,
### 14  1_time,1_type,1_collection_id,1_scheduling_class,
### 18  2_time,2_type,2_collection_id,2_scheduling_class,
### 22 3_time,3_type,3_collection_id,3_scheduling_class

# ['time','type','collection_id','scheduling_class']

# ['time','type','collection_id','scheduling_class']

# ['time','type','collection_id','scheduling_class']
class Google_cluster(torch.utils.data.Dataset):
  '''
  数据加载方式
  '''
  def __init__(self, data_path):

    # datas = pd.read_csv(data_path,usecols=[3,4,6,7,8,9,11,13,14,15,17,18,19,21])
    datas = pd.read_csv(data_path)
    dataset = []
    for index,row in datas.iterrows():
      dataset.append(row)

    # for data in datas.values:
      # dataset.append(data)

    self.dataset = dataset
  
  def __getitem__(self, i):
    # print(self.dataset[i])
    pick_data = self.dataset[i]

    # print(dd)
    
    # dd.extend(dd)
    x_data = []
    x_data.extend(get_time_list(pick_data['time']))

    x_data.append(pick_data['priority'])
    x_data.extend(get_time_list(pick_data['start_time']))
    x_data.extend(get_time_list(pick_data['end_time']))
    x_data.append(pick_data['cpus'])
    x_data.append(pick_data['memory'])
    x_data.append(pick_data['0_type'])
    x_data.append(pick_data['0_scheduling_class'])
    x_data.extend(get_time_list(pick_data['1_time']))
    x_data.append(pick_data['1_type'])
    x_data.append(pick_data['1_scheduling_class'])
    x_data.extend(get_time_list(pick_data['2_time']))
    x_data.append(pick_data['2_type'])
    x_data.append(pick_data['2_scheduling_class'])


    x_data_list = [x_data]
    for i in range(34):
      tmp = x_data[i+1:]
      tmp.extend(x_data[0:i+1])

      x_data_list.append(tmp)
      
    x = torch.tensor( np.array( x_data_list )  )

    return x.to(torch.float32),torch.tensor(int(pick_data['type'])-5)



  def __len__(self):
    return len(self.dataset)



if __name__=='__main__':

  datas = pd.read_csv('./tain.csv')
  dataset = []
  for data in datas.values:
    dataset.append(data)
  print(dataset[0].shape)
