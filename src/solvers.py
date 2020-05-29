import torch
from torch import FloatTensor, LongTensor
from torch.autograd import Variable
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import colors
import pickle
from collections import OrderedDict
from utils import expand

class TaskSolver:        

    def train(self, tasks, model_name, criterion, attention, device, n_epoch=30, lr=0.1, verbose=True):
        """
        trains the given model
        
        args:
            task: task to train the model on
            criterion: loss for train
            attention: whether to use attention mechanism
            model_name: name of the model to train
            n_epoch: number of epochs for training
            lr: learning rate for the optimization
            device: whether to use CPU or GPU
            verbose: if set to True prints additional info
        
        returns: the obtained metrics, the final predictions and the wrong ones
        """
        
        total_loss_train = []
        total_loss_val = []
        total_accuracy_val = []
        total_accuracy_train = []
        total_accuracy_val_pix = []
        total_accuracy_train_pix = []
        total_predictions = []
        total_wrong_predictions = []
        
        criterion = criterion()
        for task in tasks:
            
            sh1_big = 0
            sh2_big = 0
            for i in range(len(task['train'])):
                sh1 = task['train'][i]['input'].shape[0]
                sh2 = task['train'][i]['input'].shape[1]
                if sh1 > sh1_big:
                    sh1_big = sh1
                if sh2 > sh2_big:
                    sh2_big = sh2     
            for i in range(len(task['test'])):
                sh1 = task['test'][i]['input'].shape[0]
                sh2 = task['test'][i]['input'].shape[1]
                if sh1 > sh1_big:
                    sh1_big = sh1
                if sh2 > sh2_big:
                    sh2_big = sh2  
                    
            net = model_name(task['train'], sh1_big, sh2_big, attention).to(device)
            optimizer = Adam(net.parameters(), lr = lr)
        
            loss_train = []
            loss_val = []
            accuracy_val = []
            accuracy_train = []
            accuracy_val_pix = []
            accuracy_train_pix = []

            for epoch in tqdm(range(n_epoch)):

                net.train()
                loss_iter = 0
                for sample in task['train']:
                    img = FloatTensor(expand(sample['input'])).to(device)
                    labels = LongTensor(sample['output']).unsqueeze(dim=0).to(device)
                    optimizer.zero_grad()    
                    outputs = net(img)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()  

                net.eval()
                with torch.no_grad():

                    correct_val = 0
                    correct_val_pix = 0
                    total_val = 0
                    loss_iter_val = 0
                    predictions = []
                    wrong_pred = []
                    n_pixels_val = 0
                    for sample in task['test']:

                        img = FloatTensor(expand(sample['input'])).to(device)
                        labels = LongTensor(sample['output']).unsqueeze(dim=0).to(device)
                        outputs = net(img)
                        _, pred = torch.max(outputs.data, 1)
                        predictions.append((img, pred))  
                        n_pixels_val += pred.shape[1]*pred.shape[2]

                        total_val += labels.size(0)
                        flag =  (torch.all(torch.eq(pred, labels))).sum().item() 
                        correct_val += flag
                        if flag == 0:
                            wrong_pred.append((img, pred))
                        correct_val_pix += (torch.eq(pred, labels)).sum().item()            
                        loss = criterion(outputs, labels)
                        loss_iter_val += loss.item()

                correct_train = 0
                correct_train_pix = 0 
                total_train = 0
                loss_iter_train = 0
                n_pixels_train = 0
                for sample in task['train']:

                    img = FloatTensor(expand(sample['input'])).to(device)
                    labels = LongTensor(sample['output']).unsqueeze(dim=0).to(device)
                    outputs = net(img)
                    _, pred = torch.max(outputs.data, 1)
                    n_pixels_train += pred.shape[1]*pred.shape[2]

                    total_train += labels.size(0)
                    correct_train += (torch.all(torch.eq(pred, labels))).sum().item()
                    correct_train_pix += (torch.eq(pred, labels)).sum().item()
                    loss = criterion(outputs, labels)
                    loss_iter_train += loss.item()

                loss_train.append(loss_iter_train/len(task['train']))
                loss_val.append(loss_iter_val/len(task['test']))

                val_accuracy = 100 * correct_val / total_val
                val_accuracy_pix = 100 * correct_val_pix/(n_pixels_val)
                accuracy_val.append(val_accuracy)
                accuracy_val_pix.append(val_accuracy_pix)

                train_accuracy = 100 * correct_train / total_train
                train_accuracy_pix = 100 * correct_train_pix/(n_pixels_train)
                accuracy_train.append(train_accuracy)
                accuracy_train_pix.append(train_accuracy_pix)

                if verbose:
                    print('\nEpoch: ['+str(epoch+1)+'/'+str(n_epoch)+']')
                    print('Train loss is: {}'.format(loss_train[-1]))
                    print('Validation loss is: {}'.format(loss_val[-1]))
                    print('Train accuracy is: {} %'.format(accuracy_train[-1]))
                    print('Train accuracy for pixels is: {} %'.format(accuracy_train_pix[-1]))
                    print('Validation accuracy is: {} %'.format(accuracy_val[-1]))
                    print('Validation accuracy for pixels is: {} %'.format(accuracy_val_pix[-1]))
                    
        total_loss_train += loss_train
        total_loss_val += loss_val
        total_accuracy_train += accuracy_train
        total_accuracy_train_pix += accuracy_train_pix
        total_accuracy_val += accuracy_val
        total_accuracy_val_pix += accuracy_val_pix
        total_predictions += predictions
        total_wrong_predictions += wrong_pred

        metrics = {'loss_train': total_loss_train, 'loss_val': total_loss_val, 'accuracy_train':total_accuracy_train, 
                   'accuracy_train_pix': total_accuracy_train_pix, 'accuracy_val':total_accuracy_val, 
                   'accuracy_val_pix': total_accuracy_val_pix}
        final_pred = total_predictions
        
        return metrics, final_pred, total_wrong_predictions
    
class MetaTaskSolver:        

    def train(self, tasks, model_name, criterion, n_epoch=30, lr=0.1, device = "cpu", verbose=True,  inner_lr = 0.1, inner_iter = 10, meta_size = 50):
        """
        trains the given model
        
        args:
            task: task to train the model on
            criterion: loss for train
            model_name: name of the model to train
            n_epoch: number of epochs for training
            lr: learning rate for the optimization
            device: whether to use CPU or GPU
            verbose: if set to True prints additional info
            inner_lr: if using a meta-learning algorithm, the learning rate of the inner loop
            inner_iter: if using a meta-learning algorithm, the iterations of the inner loop
            meta_size: if using a meta-learning algorithm, how big to set the meta samples size
        
        returns: the obtained metrics, the final predictions and the wrong ones
        """
        
        total_loss_train = []
        total_loss_val = []
        total_accuracy_val = []
        total_accuracy_train = []
        total_accuracy_val_pix = []
        total_accuracy_train_pix = []
        total_predictions = []
        total_wrong_predictions = []
        
        criterion = criterion()
        
        for task in tasks:
            
            sh1_big = 0
            sh2_big = 0
            for i in range(len(task['train'])):
                sh1 = task['train'][i]['input'].shape[0]
                sh2 = task['train'][i]['input'].shape[1]
                if sh1 > sh1_big:
                    sh1_big = sh1
                if sh2 > sh2_big:
                    sh2_big = sh2     
            for i in range(len(task['test'])):
                sh1 = task['test'][i]['input'].shape[0]
                sh2 = task['test'][i]['input'].shape[1]
                if sh1 > sh1_big:
                    sh1_big = sh1
                if sh2 > sh2_big:
                    sh2_big = sh2  
        
        net = model_name(device, sh1_big, sh2_big).to(device)
        optimizer = Adam(net.parameters(), lr = lr)
        
        
        for epoch in tqdm(range(n_epoch)):

            losses = []
            for task in tasks:
            
                  loss_train = []
                  loss_val = []
                  accuracy_val = []
                  accuracy_train = []
                  accuracy_val_pix = []
                  accuracy_train_pix = []

                  inputs = []
                  outputs = []
                  for sample in task["train"]:
                   
                      inputs.append(FloatTensor(expand(sample['input'])).to(device))
                      y = LongTensor(sample["output"])
                      outputs.append(pad_crop(y, sh1_big, sh2_big, y.shape[0], y.shape[1], goal = "pad").unsqueeze(0).to(device))

                  inputs_train = inputs[:meta_size]
                  inputs_val = inputs[meta_size:]
                  outputs_train = outputs[:meta_size]
                  outputs_val = outputs[meta_size:]


                  fast_weights = OrderedDict(net.named_parameters())

                  for _ in range(inner_iter):
                      grads = []
                      loss = 0
                      for x,y in zip(inputs_train, outputs_train):
                          logits = net._forward(x.to(device), fast_weights)
                          loss += criterion(logits.to(device), y.to(device))
                      loss /= len(inputs_train)
                      gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
                      fast_weights = OrderedDict((name, param - inner_lr * grad)
                                for ((name, param), grad) in zip(fast_weights.items(), gradients))

                  loss = 0    
                  for x,y in zip(inputs_val, outputs_val):
                      logits = net._forward(x.to(device), fast_weights)
                      loss += criterion(logits.to(device), y.to(device))

                  loss /= len(inputs_val)
                  loss.backward(retain_graph=True)
                  losses.append(loss)
                  gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)

            net.train()
            optimizer.zero_grad()
            meta_loss = torch.stack(losses).mean()
            meta_loss.backward()
            optimizer.step()

            net.eval()
            with torch.no_grad():

                correct_val = 0
                correct_val_pix = 0
                total_val = 0
                loss_iter_val = 0
                predictions = []
                wrong_pred = []
                n_pixels_val = 0
                for sample in task['test']:

                    img = FloatTensor(expand(sample['input'])).to(device)
                    y = LongTensor(sample['output'])
                    labels = pad_crop(y, sh1_big, sh2_big, y.shape[0], y.shape[1], "pad").unsqueeze(0).to(device)
                    outputs = net(img)
                    _, pred = torch.max(outputs.data, 1)
                    predictions.append((img, pred))  
                    n_pixels_val += pred.shape[1]*pred.shape[2]

                    total_val += labels.size(0)
                    flag =  (torch.all(torch.eq(pred, labels))).sum().item() 
                    correct_val += flag
                    if flag == 0:
                        wrong_pred.append((img, pred))
                    correct_val_pix += (torch.eq(pred, labels)).sum().item()            
                    loss = criterion(outputs, labels)
                    loss_iter_val += loss.item()

            correct_train = 0
            correct_train_pix = 0 
            total_train = 0
            loss_iter_train = 0
            n_pixels_train = 0
            for sample in task['train']:

                img = FloatTensor(expand(sample['input'])).to(device)
                y = LongTensor(sample['output'])
                labels = pad_crop(y, sh1_big, sh2_big, y.shape[0], y.shape[1], "pad").unsqueeze(0).to(device)
                outputs = net(img)
                _, pred = torch.max(outputs.data, 1)
                n_pixels_train += pred.shape[1]*pred.shape[2]

                total_train += labels.size(0)
                correct_train += (torch.all(torch.eq(pred, labels))).sum().item()
                correct_train_pix += (torch.eq(pred, labels)).sum().item()
                loss = criterion(outputs, labels)
                loss_iter_train += loss.item()

            loss_train.append(loss_iter_train/len(task['train']))
            loss_val.append(loss_iter_val/len(task['test']))

            val_accuracy = 100 * correct_val / total_val
            val_accuracy_pix = 100 * correct_val_pix/(n_pixels_val)
            accuracy_val.append(val_accuracy)
            accuracy_val_pix.append(val_accuracy_pix)

            train_accuracy = 100 * correct_train / total_train
            train_accuracy_pix = 100 * correct_train_pix/(n_pixels_train)
            accuracy_train.append(train_accuracy)
            accuracy_train_pix.append(train_accuracy_pix)

            if verbose:
                print('\nEpoch: ['+str(epoch+1)+'/'+str(n_epoch)+']')
                print('Train loss is: {}'.format(loss_train[-1]))
                print('Validation loss is: {}'.format(loss_val[-1]))
                print('Train accuracy is: {} %'.format(accuracy_train[-1]))
                print('Train accuracy for pixels is: {} %'.format(accuracy_train_pix[-1]))
                print('Validation accuracy is: {} %'.format(accuracy_val[-1]))
                print('Validation accuracy for pixels is: {} %'.format(accuracy_val_pix[-1]))
            
            total_loss_train.append(loss_train)
            total_loss_val.append(loss_val)
            total_accuracy_train.append(accuracy_train)
            total_accuracy_train_pix.append(accuracy_train_pix)
            total_accuracy_val.append(accuracy_val)
            total_accuracy_val_pix.append(accuracy_val_pix)
            total_predictions.append(total_predictions)
            total_wrong_predictions.append(wrong_pred)
                
        metrics = {'loss_train': total_loss_train, 'loss_val': total_loss_val, 'accuracy_train':total_accuracy_train, 
                   'accuracy_train_pix': total_accuracy_train_pix, 'accuracy_val':total_accuracy_val, 
                   'accuracy_val_pix': total_accuracy_val_pix}
        final_pred = total_predictions
        
        return metrics, final_pred, total_wrong_predictions


def evaluate_metrics(ts, tasks, model_name, criterion, n_epoch, lr, device,  verbose, inner_lr = None, inner_iter = None, meta_size = 1, attention = None):
    """
    evaluates the metric of the given model on the given task
    
    args:
        ts: task solver to use
        task: task to train the model on
        criterion: loss for train
        model_name: name of the model to train
        n_epoch: number of epochs for training
        lr: learning rate for the optimization
        device: whether to use CPU or GPU
        verbose: if set to True prints additional info
        inner_lr: if using a meta-learning algorithm, the learning rate of the inner loop
        inner_iter: if using a meta-learning algorithm, the iterations of the inner loop
        meta_size: if using a meta-learning algorithm, how big to set the meta samples size
        attention: wheter to use attention mechanism in the model
    
    returns: the obtained metrics, the final predictions and the wrong ones
    """
    
    if inner_lr is not None and inner_iter is not None:
        ts = MetaTaskSolver()
        return ts.train(tasks, model_name, criterion, n_epoch, lr, device, verbose, inner_lr, inner_iter, meta_size)
    else:
        ts = TaskSolver()
        return ts.train(tasks, model_name, criterion, attention, device, n_epoch, lr, verbose)

def plot_metrics(train_result):
    """
    plots the metrics obtained
    """
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(35,7), dpi=50)

    ax1.plot(train_result['loss_train'], c='r', label = 'Train loss')
    ax1.plot(train_result['loss_val'], c='g', label = 'Val. loss')
    ax1.legend(prop={'size': 18}, loc='upper right')
    ax1.set_xlabel('Loss', fontsize=22)
    ax1.tick_params(labelsize=18)

    ax2.plot(train_result['accuracy_train'], c='r', label = 'Train accuracy')
    ax2.plot(train_result['accuracy_val'], c='g', label = 'Val. accuracy')
    ax2.legend(prop={'size': 18}, loc='lower right')
    ax2.set_xlabel('Accuracy', fontsize=22)
    ax2.tick_params(labelsize=18)

    ax3.plot(train_result['accuracy_train_pix'], c='r', label = 'Train accuracy on pixels')
    ax3.plot(train_result['accuracy_val_pix'], c='g', label = 'Val. accuracy on pixels')
    ax3.legend(prop={'size': 18}, loc='lower right')
    ax3.set_xlabel('Accuracy for pixels', fontsize=22)
    ax3.tick_params(labelsize=18)

    plt.show()

def compare_plots(results, n = 4):
    """
    plots predictions obtained and compare to the input
    """
    
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)
    fig, axs = plt.subplots(2, n, figsize=(4*n,8), dpi=50)

    for i in range(n):
      _, original = torch.max(results[i][0], 0)
      original = original.cpu()
      predicted = results[i][1].squeeze(0).cpu()
      axs[0][i].imshow(original, cmap=cmap, norm=norm)
      axs[0][i].set_title('Original')
      axs[1][i].imshow(predicted, cmap=cmap, norm=norm)
      axs[1][i].set_title('Predicted')

    plt.show()

def find_correct_pred(pred_all, pred_wrong):
    """
    return a list of correct predictions
    """
    correct = []
    for t in pred_all:
      t_check = np.array(t[1].squeeze(0).cpu())
      for t_wrong in pred_wrong:
        t_wrong_check = np.array(t_wrong[1].squeeze(0).cpu())
        if (t_check != t_wrong_check).all():
          correct.append(t)
    return correct

def save_results(path, list_res, list_names):
    """
    save results output from evaluate function.
    
    args:
        path: path to directory in which save files
        list_res: list including the three output of evaluate function
        list_names: list of names to give to each output file in list_res  
    """
    
    with open(path+list_names[0]+'.pickle', 'wb') as handle:
      pickle.dump(list_res[0], handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(path+list_names[1]+'.pickle', 'wb') as handle:
      pickle.dump(list_res[1], handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(path+list_names[2]+'.pickle', 'wb') as handle:
      pickle.dump(list_res[2], handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_results(path, filename):
    """
    load result from pickle files.
  
    args:
        path: path to directory in which file is located
        filename: name of the file (without pickle extention) 
    """  
    with open(path+filename+'.pickle', 'rb') as handle:
      return pickle.load(handle)
