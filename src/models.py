import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import pad_crop

#----------------------------------_____________________ATTENTION___________________-----------------------------------  

class Attention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(Attention, self).__init__()
        self.linear = nn.Linear(input_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)

    def forward(self, x):
        att = self.linear(x)
        att = self.full_att(F.relu(x.unsqueeze(1))).squeeze(2) 
        alpha = F.softmax(att)
        out = (x * alpha.unsqueeze(2)).sum(dim=1)
        return x
    

#----------------------------------_____________________DENSE___________________----------------------------------- 

class FullyCon(nn.Module):
    """
    Fully connected
    """
  
    def __init__(self, task_train, w, h, attention):
        super(FullyCon, self).__init__()

        self.inp_dim = np.array(np.array(task_train[0]['input']).shape)
        self.out_dim = np.array(np.array(task_train[0]['output']).shape)
        self.linear = nn.Linear(self.inp_dim[0]*self.inp_dim[1]*10, self.out_dim[0]*self.out_dim[1]*10)

    def forward(self, x):
        x = x.view(-1) 
        x = self.linear(x) 
        x = torch.reshape(x, (1, 10, self.out_dim[0], self.out_dim[1]))
        return x
    
class FullyCon_deep(nn.Module):
    """
    Fully connected
    """
  
    def __init__(self, task_train, w, h, attention):
        super(FullyCon_deep, self).__init__()

        self.inp_dim = np.array(np.array(task_train[0]['input']).shape)
        self.out_dim = np.array(np.array(task_train[0]['output']).shape)
        self.linear = nn.Linear(self.inp_dim[0]*self.inp_dim[1]*10, self.out_dim[0]*self.out_dim[1]*10)
        self.linear2 = nn.Linear(self.out_dim[0]*self.out_dim[1]*10, self.out_dim[0]*self.out_dim[1]*10)
        
    def forward(self, x):
        x = x.view(-1) 
        x = self.linear(x)
        x = F.leaky_relu(x)
        x = self.linear2(x)
        x = torch.reshape(x, (1, 10, self.out_dim[0], self.out_dim[1]))
        return x
    
    
    
class FullyCon_T4(nn.Module):
    """
    Fully connected for the 4th task
    """
    
    def __init__(self, task_train, w, h, attention):
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
    
#----------------------------------_____________________CNN___________________----------------------------------- 
    
    
class CNN_5(nn.Module):
    """
    CNN
    """
    
    def __init__(self, task_train, w, h, attention):
        super(CNN_5, self).__init__()

        self.inp_dim = np.array(np.array(task_train[0]['input']).shape)
        self.out_dim = np.array(np.array(task_train[0]['output']).shape)
        self.conv = nn.Conv2d(in_channels=10, out_channels=100, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=100, out_channels=10, kernel_size=5, padding=2)
        self.attention = Attention(10, 10)
        self.attention_value = attention

    def forward(self, x):  
        ch, input_w, input_h = x.shape[0], x.shape[1], x.shape[2]
        x = pad_crop(x, self.out_dim[0], self.out_dim[1], self.inp_dim[0], self.inp_dim[1], goal='pad')
        x = x.unsqueeze(0)
        #x = torch.nn.Upsample(size=(self.out_dim[0], self.out_dim[1]))(x)
        x = self.conv(x)
        x =self.conv2(x)
        if self.attention_value is not None:
            x = torch.reshape(x, (1, x.shape[2], x.shape[3], 10))
            x = self.attention(x)
            x = torch.reshape(x, (1, 10, x.shape[1], x.shape[2]))
        return x
    
class CNN_10(nn.Module):
    """
    CNN
    """
    
    def __init__(self, task_train, w, h, attention):
        super(CNN_10 , self).__init__()

        self.inp_dim = np.array(np.array(task_train[0]['input']).shape)
        self.out_dim = np.array(np.array(task_train[0]['output']).shape)
        self.conv = nn.Conv2d(in_channels=10, out_channels=100, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=100, out_channels=10, kernel_size=3, padding=1)
        self.attention = Attention(10, 10)
        self.attention_value = attention

    def forward(self, x):  
        ch, input_w, input_h = x.shape[0], x.shape[1], x.shape[2]
        x = pad_crop(x, self.out_dim[0], self.out_dim[1], self.inp_dim[0], self.inp_dim[1], goal='pad')
        x = x.unsqueeze(0)
        #x = torch.nn.Upsample(size=(self.out_dim[0], self.out_dim[1]))(x)
        x = self.conv(x)
        x =self.conv2(x)
        if self.attention_value is not None:
            x = torch.reshape(x, (1, x.shape[2], x.shape[3], 10))
            x = self.attention(x)
            x = torch.reshape(x, (1, 10, x.shape[1], x.shape[2]))
        return x
    
    

#----------------------------------_____________________LSTM___________________----------------------------------- 


class LSTM(nn.Module):
    """
    LSTM
    """
    
    def __init__(self, task_train, w, h, attention):
        super(LSTM, self).__init__()

        self.inp_dim = np.array(np.array(task_train[0]['input']).shape)
        self.out_dim = np.array(np.array(task_train[0]['output']).shape)
        self.lstm = nn.LSTM(self.inp_dim[0]*self.inp_dim[1], self.out_dim[0]*self.out_dim[1], dropout=0, batch_first=True)
        self.attention = Attention(10, 10)
        self.attention_value = attention

    def forward(self, x): 
        sh1, sh2, sh3 = x.shape[0], x.shape[1], x.shape[2]
        x = x.view(sh1, -1).unsqueeze(0)
        x, _ = self.lstm(x)
        if self.attention_value is not None:
            x = torch.reshape(x, (1, x.shape[2], 10))
            x = self.attention(x)
        x = torch.reshape(x, (1, 10, self.out_dim[0], self.out_dim[1])) 
        return x

class LSTM_T4(nn.Module):
    """
    LSTM for the 4th task
    """
    def __init__(self, task_train, w, h, attention):
        super(LSTM_T4, self).__init__() 

        self.largest_w = w
        self.largest_h = h
        self.lstm = nn.LSTM(self.largest_w*self.largest_h, self.largest_w*self.largest_h, dropout=0, batch_first=True)
        self.attention = Attention(10, 10)
        self.attention_value = attention
        
    def forward(self, x):     
        ch, input_w, input_h = x.shape[0], x.shape[1], x.shape[2]
        x = pad_crop(x, self.largest_w, self.largest_h, input_w, input_h, goal='pad')
        x = x.view(ch, -1).unsqueeze(0)
        x, _ = self.lstm(x)
        if self.attention_value is not None:
            x = torch.reshape(x, (1, x.shape[2], 10))
            x = self.attention(x)
        x = torch.reshape(x, (1, 10, self.largest_w, self.largest_h)) 
        x = pad_crop(x, input_w, input_h, self.largest_w, self.largest_h, goal='crop')
        return x

    
    
#----------------------------------_____________________LSTM___________________----------------------------------- 


class GRU(nn.Module):
    """
    GRU
    """
    
    def __init__(self, task_train, w, h, attention):
        super(GRU, self).__init__()

        self.inp_dim = np.array(np.array(task_train[0]['input']).shape)
        self.out_dim = np.array(np.array(task_train[0]['output']).shape)
        self.gru = nn.GRU(self.inp_dim[0]*self.inp_dim[1], self.out_dim[0]*self.out_dim[1], dropout=0, batch_first=True)
        self.attention = Attention(10, 10)
        self.attention_value = attention

    def forward(self, x): 
        sh1, sh2, sh3 = x.shape[0], x.shape[1], x.shape[2]
        x = x.view(sh1, -1).unsqueeze(0)
        x, _ = self.gru(x)
        if self.attention_value is not None:
            x = torch.reshape(x, (1, x.shape[2], 10))
            x = self.attention(x)
        x = torch.reshape(x, (1, 10, self.out_dim[0], self.out_dim[1])) 
        return x

class LSTM_T4(nn.Module):
    """
    LSTM for the 4th task
    """
    def __init__(self, task_train, w, h, attention):
        super(LSTM_T4, self).__init__() 

        self.largest_w = w
        self.largest_h = h
        self.lstm = nn.LSTM(self.largest_w*self.largest_h, self.largest_w*self.largest_h, dropout=0, batch_first=True)
        self.attention = Attention(10, 10)
        self.attention_value = attention
        
    def forward(self, x):     
        ch, input_w, input_h = x.shape[0], x.shape[1], x.shape[2]
        x = pad_crop(x, self.largest_w, self.largest_h, input_w, input_h, goal='pad')
        x = x.view(ch, -1).unsqueeze(0)
        x, _ = self.lstm(x)
        if self.attention_value is not None:
            x = torch.reshape(x, (1, x.shape[2], 10))
            x = self.attention(x)
        x = torch.reshape(x, (1, 10, self.largest_w, self.largest_h)) 
        x = pad_crop(x, input_w, input_h, self.largest_w, self.largest_h, goal='crop')
        return x

    
#----------------------------------_____________________META LEARNING___________________-----------------------------------    
    
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

class BasicBlock(nn.Module):
    """
    BasicBlock implementation for ResNet
    
    reference: https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
    """
    
    expansion = 1

    def __init__(self, device, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.device = device
        self.stride = stride
        self.in_planes = in_planes
        self.planes = planes
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, 
                              stride=stride, padding=2, bias=False)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False)
            )

    def forward(self,x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
    def _forward(self, n, i, weights, x):
        out = F.conv2d(x, weights["layer{}.{}.conv1.weight".format(n,i)], stride = self.stride,padding = 2)
        out = F.relu(out)
        out = F.conv2d(out, weights["layer{}.{}.conv2.weight".format(n,i)])
        conv = 0
        if self.stride != 1 or self.in_planes != self.expansion*self.planes:
            
            conv = F.conv2d(x, weights["layer{}.{}.shortcut.0.weight".format(n,i)],stride = self.stride)
            x += conv
                                
        out = F.relu(out)
        return out
    
    

class ResNet(nn.Module):
    """
    ResNet implementation
    
    reference: https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
    """
    
    def __init__(self,device, w, h, block = BasicBlock, num_blocks = [1,1,1,1], num_classes=10):
      
        super(ResNet, self).__init__()
        self.in_planes = 8
        self.largest_w = w
        self.largest_h = h
        self.device = device

        self.conv1 = nn.Conv2d(10, 8, kernel_size=3, padding = 1,stride=1, bias=False)
        self.layer1 = self._make_layer(1, block, 8, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(2, block, 8, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(3, block, 8, num_blocks[2], stride=1)
        self.layer4 = self._make_layer(4, block, 8, num_blocks[3], stride=1)
        self.linear = nn.Linear(8*w*h, 10*w*h)


    def _make_layer(self, n, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.device, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
   
    def forward(self, x):
        ch, input_w, input_h = x.shape[0], x.shape[1], x.shape[2]
        x = pad_crop(x, self.largest_w, self.largest_h, input_w, input_h, goal='pad')
        x = x.unsqueeze(0)
        
        out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        x = torch.reshape(out, (1, 10, self.largest_w,self.largest_h )) 
        return x
     
    def _forward(self, x, weights):
        ch, input_w, input_h = x.shape[0], x.shape[1], x.shape[2]
        x = pad_crop(x, self.largest_w, self.largest_h, input_w, input_h, goal='pad')
        x = x.unsqueeze(0)
        
        out = F.conv2d(x, weights["conv1.weight"], padding = 1)
        out = F.relu(out)
        
        for i, layer in enumerate(self.layer1):
            out = layer._forward(1, i, weights, out)
        for i, layer in enumerate(self.layer2):
            out = layer._forward(2, i, weights, out)
        for i, layer in enumerate(self.layer3):
            out = layer._forward(3, i, weights, out)
        for i, layer in enumerate(self.layer4):
            out = layer._forward(4, i, weights, out)

        out = out.view(out.size(0), -1)
        out = F.linear(out, weights["linear.weight"], weights["linear.bias"])
        
        x = torch.reshape(out, (1, 10, self.largest_w, self.largest_h)) 
        return x