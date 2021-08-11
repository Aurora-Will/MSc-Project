
import pandas as pd
from collections import Counter
import numpy as np

####总的数据量{7: 28208820, 6: 480911, 5: 143261}


def save_types():
  chunksize = 100000
  total_keys = {}
  root_data = "D:/ali/source_data.csv"
  max_index = 0
  start=np.array([])
  for df in pd.read_csv(root_data,chunksize=chunksize):
      # for index,row in df.iterrows():
      #     # print(count,row)  
      #     if index >max_index:
      #       max_index = index

      print(df.index.values)

      type_np = df['type'].values
      start = np.concatenate((start,type_np),axis=0)

  np.save('types.npy',start)

  #     tmp_list = df['type'].tolist()
  #     result = Counter(tmp_list)

  #     for key in result.keys():
  #       if not key in total_keys.keys():
  #         total_keys[key] = 0

  #       total_keys[key] += result[key]
        
  #     # print( dict(result))

  # print(max_index)
  # print(total_keys)


def pick_index():
  import random
  random.seed(10)
  types = np.load('types.npy')
  pick_dict_list ={5:[],6:[],7:[]}
  for idx,t in enumerate(types) :
    pick_dict_list[t].append(idx)

  for key in pick_dict_list.keys():
    random.shuffle(pick_dict_list[key])

  train_pick_index_list = []
  val_pick_index_list = []


  for key in pick_dict_list.keys():
    train_pick_index_list.extend(pick_dict_list[key][:10000])
    val_pick_index_list.extend(pick_dict_list[key][10000:13000])

  train_pick_index_list.sort()
  val_pick_index_list.sort()

  np.save('train_index.npy',np.array(train_pick_index_list))
  np.save('val_index.npy',np.array(val_pick_index_list))

  print(types.shape)



def create_train_val():
  train_indexs = np.load('train_index.npy')
  val_indexs = np.load('val_index.npy')
  train_indexs = train_indexs.tolist()
  val_indexs = val_indexs.tolist()
  chunksize = 100000
  root_data = "D:/ali/source_data.csv"

  train_data = []
  val_data =  []

  current_circle = 0

  for df in pd.read_csv(root_data,chunksize=chunksize):
      exist_v = []
      for idx,train_index in enumerate(train_indexs) :
        left = train_index%chunksize
        circle = int(train_index/chunksize)

        if circle == current_circle:
          tmp_df = df.iloc[left:left+1]
          train_data.append(tmp_df)
          exist_v.append(train_index)
        else:
          continue
      
      for v in exist_v:
        train_indexs.remove(v)


      #########val
      exist_v = []
      for idx,val_index in enumerate(val_indexs) :
        left = val_index%chunksize
        circle = int(val_index/chunksize)

        if circle == current_circle:
          tmp_df = df.iloc[left:left+1]
          val_data.append(tmp_df)
          exist_v.append(val_index)
        else:
          continue
      
      for v in exist_v:
        val_indexs.remove(v)
      current_circle +=1



  train_df = train_data[0]

  for tmp_df in train_data[1:]:
    train_df = pd.concat([train_df,tmp_df],axis=0)
  train_df.to_csv('./tain.csv',index = False)



  val_df = val_data[0]
  for tmp_df in val_data[1:]:
    val_df = pd.concat([val_df,tmp_df],axis=0)
  val_df.to_csv('./val.csv',index = False)


def create_data():
  one_name='a'
  two_name='b'
  type_v = 5

  # one_name='c'
  # two_name='d'
  # type_v = 6


  one_name='e'
  two_name='f'
  type_v = 7


  collection_df_path = "./data/"+one_name+".csv"
  collection_df = pd.read_csv(collection_df_path)
  collection_tasks_path = "./data/"+two_name+".csv"
  collection_tasks = pd.read_csv(collection_tasks_path)
  collection_tasks_list = []

  for index,row in collection_tasks.iterrows():
    tmp_data = {}
    for key in ['time','type','collection_id','scheduling_class']:
      tmp_data[key] = row[key]
    collection_tasks_list.append(tmp_data)

  collection_tasks_list.sort(key=lambda k: (k.get('collection_id', 0), k.get('time', 0)))

  ##########整理每个collection_id
  # print(collection_tasks_list[:8])
  a = collection_tasks_list[0]
  collection_tasks_dict = {}
  count_num = 1
  count_list = []
  record_collection_id = int(a['collection_id'])
  # print(record_collection_id)
  collection_tasks_dict[record_collection_id] = [collection_tasks_list[0]]

  # print(collection_tasks_dict)

  #####只需要四个状态扭转的collection  
  for tmp in collection_tasks_list[1:]:
    if tmp['collection_id'] ==record_collection_id:

      count_num +=1
    else:
      count_list.append(count_num)
      count_num =1
      record_collection_id = int(tmp['collection_id'])
      collection_tasks_dict[record_collection_id] = []


    collection_tasks_dict[record_collection_id].append(tmp)


  from collections import Counter
  result = Counter(count_list)
  print (result)

  count_collection = 0
  keys = ['time','instance_index','collection_id','type','priority','machine_id','start_time','end_time','cpus','memory' ]
  ddddddd = 0
  total_data = []
  for index,row in collection_df.iterrows():
    if row['type'] ==type_v:
      tmpdata = []
      for key in keys:
        tmpdata.append(row[key])
      
      collection_id_int = int(row['collection_id'])

      try:
        if len(collection_tasks_dict[collection_id_int])==4:
          for i in range(4):
            for task_key in ['time','type','collection_id','scheduling_class']:
              tmpdata.append(collection_tasks_dict[collection_id_int][i][task_key])

        if len(tmpdata)==26:
          count_collection +=1
          total_data.append(tmpdata)

      except Exception as e:
        # print(collection_id_int)
        ddddddd +=1

        pass

  print(ddddddd)

  columns = keys
  for i in range(4):
    for task_key in ['time','type','collection_id','scheduling_class']:
      columns.append(str(i)+"_"+task_key)

  df = pd.DataFrame(total_data,columns=columns) 
  print(df.shape)
  df.to_csv("./data/"+str(type_v)+".csv",index=False)



# create_data()
data_7 = pd.read_csv('./data/7.csv')
data_7 =data_7.iloc[:30000]
data_6 = pd.read_csv('./data/6.csv')
data_6 =data_6.iloc[:30000]
data_5 = pd.read_csv('./data/5.csv')
data_5 =data_5.iloc[:30000]


train_df = pd.concat([pd.concat([data_7.iloc[:21000],data_6.iloc[:21000]],axis=0),data_5.iloc[:21000]],axis=0) 
train_df.to_csv('./tain.csv',index = False)
val_df = pd.concat([pd.concat([data_7.iloc[21000:],data_6.iloc[21000:]],axis=0),data_5.iloc[21000:]],axis=0) 
val_df.to_csv('./val.csv',index = False)






