import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import pad_crop

class LSTM(nn.Module):
    """
    LSTM
    """
    
    def __init__(self, task_train, w, h):
        super(LSTM, self).__init__()

        self.inp_dim = np.array(np.array(task_train[0]['input']).shape)
        self.out_dim = np.array(np.array(task_train[0]['output']).shape)
        self.lstm = nn.LSTM(self.inp_dim[0]*self.inp_dim[1], self.out_dim[0]*self.out_dim[1], dropout=0, batch_first=True)

    def forward(self, x): 
        sh1, sh2, sh3 = x.shape[0], x.shape[1], x.shape[2]
        x = x.view(sh1, -1).unsqueeze(0)
        x, _ = self.lstm(x)
        x = torch.reshape(x, (1, 10, self.out_dim[0], self.out_dim[1])) 
        return x
    

class CNN(nn.Module):
    """
    CNN
    """
    
    def __init__(self, task_train, w, h):
        super(CNN, self).__init__()

        self.out_dim = np.array(np.array(task_train[0]['output']).shape)
        self.conv = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=5, padding=2)

    def forward(self, x):  
        x = x.unsqueeze(0)
        x = torch.nn.Upsample(size=(self.out_dim[0], self.out_dim[1]))(x)
        x = self.conv(x)
        return x
    
class FullyCon(nn.Module):
    """
    Fully connected
    """
  
    def __init__(self, task_train, w, h):
        super(FullyCon, self).__init__()

        self.inp_dim = np.array(np.array(task_train[0]['input']).shape)
        self.out_dim = np.array(np.array(task_train[0]['output']).shape)
        self.linear = nn.Linear(self.inp_dim[0]*self.inp_dim[1]*10, self.out_dim[0]*self.out_dim[1]*10)

    def forward(self, x):
        x = x.view(-1) 
        x = self.linear(x) 
        x = torch.reshape(x, (1, 10, self.out_dim[0], self.out_dim[1]))
        return x
    
class MetaFullyCon(torch.nn.Module):
    """
    Fully connected for meta learning
    """
    
    def __init__(self, task_train, w, h):
        super(MetaFullyCon, self).__init__()
        self.largest_w = w
        self.largest_h = h
        self.out_dim = np.array(np.array(task_train[0]['output']).shape)
        self.lin = nn.Linear(10*self.largest_w*self.largest_h, 10*self.out_dim[0]*self.out_dim[1])

    def forward(self, x):
        ch, input_w, input_h = x.shape[0], x.shape[1], x.shape[2]
        x = pad_crop(x, self.largest_w, self.largest_h, input_w, input_h, goal='pad')
        x = torch.flatten(x)
        x = self.lin(x)
        x = x.reshape((1,10,self.out_dim[0],self.out_dim[1]))
        return x

    def _forward(self, x, weights):
        ch, input_w, input_h = x.shape[0], x.shape[1], x.shape[2]
        x = pad_crop(x, self.largest_w, self.largest_h, input_w, input_h, goal='pad')
        x = torch.flatten(x)
        x = F.linear(x, weights['lin.weight'], bias = weights['lin.bias'])
        x = x.reshape((1,10,self.out_dim[0],self.out_dim[1]))
        return x
    
class FullyCon_T4(nn.Module):
    """
    Fully connected for the 4th task
    """
    
    def __init__(self, task_train, w, h):
        super(FullyCon_T4, self).__init__()

        self.largest_w = w
        self.largest_h = h
        self.linear = nn.Linear(self.largest_w*self.largest_h*10, self.largest_w*self.largest_h*10)

    def forward(self, x):  
        ch, input_w, input_h = x.shape[0], x.shape[1], x.shape[2]
        x = pad_crop(x, self.largest_w, self.largest_h, input_w, input_h, goal='pad') 
        x = x.view(-1).unsqueeze(0) 
        x = self.linear(x) 
        x = torch.reshape(x, (1, 10, self.largest_w, self.largest_h)) 
        x = pad_crop(x, input_w, input_h, self.largest_w, self.largest_h, goal='crop')
        return x
    
class LSTM_T4(nn.Module):
    """
    LSTM for the 4th task
    """

    def __init__(self, task_train, w, h):
        super(LSTM_T4, self).__init__() 

        self.largest_w = w
        self.largest_h = h
        self.LSTM = nn.LSTM(self.largest_w*self.largest_h, self.largest_w*self.largest_h, dropout=0, batch_first=True)
        
    def forward(self, x):     
        ch, input_w, input_h = x.shape[0], x.shape[1], x.shape[2]
        x = pad_crop(x, self.largest_w, self.largest_h, input_w, input_h, goal='pad')
        x = x.view(ch, -1).unsqueeze(0)
        x, _ = self.LSTM(x)
        x = torch.reshape(x, (1, 10, self.largest_w, self.largest_h)) 
        x = pad_crop(x, input_w, input_h, self.largest_w, self.largest_h, goal='crop')
        return x
    
class CNN_T4(nn.Module):
    """
    CNN for the 4th task
    """
    
    def __init__(self, task_train, sh1_big, sh2_big):
        super(CNN_T4, self).__init__()

        self.largest_w = sh1_big
        self.largest_h = sh2_big
        self.conv = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=5, padding=2)

    def forward(self, x):  
        ch, input_w, input_h = x.shape[0], x.shape[1], x.shape[2]
        x = pad_crop(x, self.largest_w, self.largest_h, input_w, input_h, goal='pad')
        x = x.unsqueeze(0)
        x = self.conv(x)
        x = x = pad_crop(x, input_w, input_h, self.largest_w, self.largest_h, goal='crop')
        return x