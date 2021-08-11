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


class Net(nn.Module):
  def __init__(self,cols,size_hidden,classes):
      super(Net, self).__init__()
      #Note that 17 is the number of columns in the input matrix. 
      self.fc1 = torch.nn.Sequential(
        torch.nn.Linear(cols, size_hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(size_hidden, size_hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(size_hidden, size_hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(size_hidden, size_hidden*2),
        torch.nn.Linear(size_hidden*2, size_hidden),
    )

      self.fc2 = nn.Linear(size_hidden, classes)
      
  def forward(self, x):
      x = self.fc1(x)

      x = self.fc2(x)
      return x

def init_model(cols):
    size_hidden = 1000
    classes = 3
    net = Net(cols, size_hidden, classes)


    return net





def main():

        
    log_dir = os.path.join('tensorboard', 'train')
    os.makedirs(log_dir,exist_ok=True)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    writer = SummaryWriter(log_dir=log_dir)

    train_np_path = "./tain.csv"
    val_np_path = "./val.csv"
    train_data=Google_cluster(train_np_path) 
    val_data=Google_cluster(val_np_path)
    batch_size = 2000*15
    train_loader = torch.utils.data.DataLoader(train_data,batch_size = batch_size,shuffle=True)
    valid_loader = torch.utils.data.DataLoader(val_data,batch_size = batch_size)

    # net = init_model(5)

    net = MSResNet(input_channel=30, layers=[1, 1, 1, 1], num_classes=3)
    # print(net)
    # exec()
# msresnet = msresnet.cuda()


    net.to(device)

    num_epochs = 300

    params = filter(lambda p: p.requires_grad, net.parameters())#####训练的参量

    lr=0.01 
    t=int(num_epochs/5)#warmup
    T=num_epochs#共有120个epoch，则用于cosine rate的一共有110个epoch
    n_t=0.5
    optimizer = optim.Adam(params, lr=lr)#,weight_decay=0

    lambda1 = lambda epoch: (0.9*epoch / t+0.001) if epoch < t else  0.001  if n_t * (1+math.cos(math.pi*(epoch - t)/(T-t)))<0.001 else n_t * (1+math.cos(math.pi*(epoch - t)/(T-t)))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda1)


    scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=10,T_mult=2)

    

    save_path = "./model.pth"
    # define loss function
    criterion = nn.CrossEntropyLoss()
    best_val_loss = None
    
    for epoch in range(num_epochs):
        e = epoch
        # train
        net.train()
        running_loss = 0.0
        accuracy_train = 0
        start_time = time.time()
        for inputs,labels in train_loader:
            if torch.cuda.is_available():
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            # print(inputs.size())
            optimizer.zero_grad()
            logits = net.forward(inputs)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            ps = torch.exp(logits)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy_train += equality.type(torch.FloatTensor).mean()
            # print statistics
            running_loss += loss.item()

        scheduler.step()
        # validate
        net.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_accuracy = 0

            for inputs, labels in valid_loader:
                if torch.cuda.is_available() :
                    inputs, labels = inputs.to('cuda'), labels.to('cuda')
                
                outputs = net.forward(inputs)
                val_loss += criterion(outputs, labels).item()

                ps = torch.exp(outputs)
                equality = (labels.data == ps.max(dim=1)[1])
                val_accuracy += equality.type(torch.FloatTensor).mean()
        end_time = time.time() 
        print(end_time - start_time)


        writer.add_scalar('Train/Loss', running_loss/len(train_loader),e+1)
        writer.add_scalar('Train/Acc',accuracy_train/len(train_loader),e+1)
        writer.add_scalar('Validation/Loss',val_loss/len(valid_loader),e+1)
        writer.add_scalar('Validation/Acc',val_accuracy/len(valid_loader),e+1)

        print(f"Epoch {e+1}/{num_epochs}.. "
                f"Training loss: {running_loss/len(train_loader):.3f}.. "
                f"Training accuracy: {accuracy_train/len(train_loader):.3f}  "
                f"Validation loss: {val_loss/len(valid_loader):.3f}.. "
                f"Validation accuracy: {val_accuracy/len(valid_loader):.3f}")

        val_loss_mean = val_loss/len(valid_loader)
        if best_val_loss is None or (val_loss_mean <best_val_loss):
            best_val_loss = val_loss_mean
            torch.save(net.state_dict(), save_path)
            
    writer.close()
    print('Finished Training')


if __name__ == '__main__':
    main()