
import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
import time
import os
import math
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import matplotlib.pyplot as plt
import numpy as np
#Define the model 
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_load import Google_cluster
from model import MSResNet
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
import numpy as np

def softmax(x):
    """
    softmax归一化计算
    :param x: 大小为(m, k)
    :return: 大小为(m, k)
    """
    # 数值稳定性考虑，减去最大值
    x -= np.atleast_2d(np.max(x, axis=1)).T
    exps = np.exp(x)
    return exps / np.atleast_2d(np.sum(exps, axis=1)).T


log_dir = os.path.join('tensorboard', 'train')
os.makedirs(log_dir,exist_ok=True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))
writer = SummaryWriter(log_dir=log_dir)

train_np_path = "./tain.csv"
val_np_path = "./val.csv"
train_data=Google_cluster(train_np_path) 
val_data=Google_cluster(val_np_path)
batch_size = 1
# valid_loader = torch.utils.data.DataLoader(val_data,batch_size = batch_size)

net = MSResNet(input_channel=30, layers=[1, 1, 1, 1], num_classes=3)
net.to(device)

net.load_state_dict(torch.load('./a/sss.pth'))
net.eval()

params = filter(lambda p: p.requires_grad, net.parameters())#####训练的参量


num_epochs = 300

lr=0.01 
t=int(num_epochs/5)#warmup
T=num_epochs#共有120个epoch，则用于cosine rate的一共有110个epoch
n_t=0.5
optimizer = optim.Adam(params, lr=lr)#,weight_decay=0

scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=10,T_mult=2)
lrs=[]
x =[]
for epoch in range(num_epochs):
    scheduler.step()
    dd = scheduler.get_lr()
    lrs.append(dd[0])
    x.append(epoch+1)
    # print(dd)

plt.title('Train Model Analysis')
# plt.plot(train_acc_x, train_acc_y, color='green', label='train accuracy')
# plt.plot(val_acc_x, val_acc_y, color='red', label='val accuracy')

plt.plot(x, lrs, color='red', label='lr')
plt.legend() # 显示图例
  
plt.xlabel('epochs')
plt.ylabel('lr')
plt.show()
exec()


# y_test=[]
# y_score = []
# with torch.no_grad():
#     val_loss = 0.0
#     val_accuracy = 0

#     for inputs, labels in valid_loader:
#         if torch.cuda.is_available() :
#             inputs= inputs.to('cuda')
#             # , labels.to('cuda')
        
#         outputs = net.forward(inputs)


#         label_v = labels.cpu().numpy()[0]
#         scores = softmax(outputs.cpu().numpy())

#         if label_v == 0:
#           y_test.append([1,0,0])
#         if label_v == 1:
#           y_test.append([0,1,0])
#         if label_v == 2:
#           y_test.append([0,0,1])

#         y_score.append(scores[0])

# np.save('val_label.npy',np.array(y_test))
# np.save('val_score.npy',np.array(y_score))

y_test= np.load('val_label.npy')
y_score = np.load('val_score.npy')
n_classes =3
# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw=2
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

# plt.plot(fpr["macro"], tpr["macro"],
#          label='macro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["macro"]),
#          color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

# plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC')
plt.legend(loc="lower right")
plt.show()