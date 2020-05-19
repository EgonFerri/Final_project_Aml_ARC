import numpy as np              #numpy library is used to work with multidimensional array.
import pandas as pd             #panda used for data manipulation and analysis.
                 
import os                       #os library is used for loading file to use in the program
import json                     #json library parses json into a string or dict, and convert string or dict to json file.
from pathlib import Path        #support path

import matplotlib.pyplot as plt #support ploting a figure
from matplotlib import colors   #colors support converting number or argument into colors


# get the path for training_task, evaluation_task, and test_task
data_path = Path('./data')
training_path = data_path / 'training'
evaluation_path = data_path / 'evaluation'
test_path = data_path / 'test'

#from the path above, we load the tests file's directory into our training_tasks, evaluation_tasks, and test_tasks variables
#the sorted() function is just for the list of directory to maintain some order
training_tasks = sorted(os.listdir(training_path))
evaluation_tasks = sorted(os.listdir(evaluation_path))
test_tasks = sorted(os.listdir(test_path))

def get_task(subset='train', index=0, print_path=False):
    if subset=='train':
        task_file = str(training_path / training_tasks[index])
    elif subset=='eval':
        task_file = str(evaluation_path / evaluation_tasks[index])
    else:
        task_file = str(test_path / test_tasks[index])
    with open(task_file, 'r') as f:   
        task = json.load(f)
    if print_path==True:
        print(task_file)
    return(task)


cmap = colors.ListedColormap(
    ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)

#plotting the training task and the test task.
def plot_task(task):
    n = len(task["train"]) + len(task["test"])
    fig, axs = plt.subplots(2, n, figsize=(4*n,8), dpi=50)
    plt.subplots_adjust(wspace=0, hspace=0)
    fig_num = 0
    for i, t in enumerate(task["train"]):
        t_in, t_out = np.array(t["input"]), np.array(t["output"])
        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)
        axs[0][fig_num].set_title(f'Train-{i} in')
        #axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))
        #axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))
        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)
        axs[1][fig_num].set_title(f'Train-{i} out')
        #axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))
        #axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))
        fig_num += 1
    for i, t in enumerate(task["test"]):
        t_in, t_out = np.array(t["input"]), np.array(t.get('output', [[0,0],[0,0]]))
        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)
        axs[0][fig_num].set_title(f'Test-{i} in')
        #axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))
        #axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))
        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)
        axs[1][fig_num].set_title(f'Test-{i} out')
        #axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))
        #axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))
        fig_num += 1
    
    plt.tight_layout()
    plt.show()
    
    
def check_dim(dataset='train'):
    type_1=[]
    type_1_1=[]
    type_1_2=[]
    type_1_3=[]
    type_1_4=[]
    
    type_2=[]
    type_2_1=[]
    type_2_2=[]
    type_2_3=[]
    
    leng=400
    if dataset=='test':
        leng=100
    for t in range(0, leng):
        task=get_task(dataset, t)
        inp=[]
        out=[]
        for i in task['train']:
            inp.append(np.array(i['input']).shape)
            out.append(np.array(i['output']).shape)
        if all(x==inp[0] for x in inp):
            type_1.append(t)
            if all(x==inp[0] for x in out):
                type_1_1.append(t)
            else:
                if all(x==out[0] for x in out):
                    if (out[0][0]*out[0][1])<(inp[0][0]*inp[0][1]):     
                        type_1_2.append(t)
                    else:
                        type_1_3.append(t)
                else:
                    type_1_4.append(t)
        else:
            type_2.append(t)
            if all(inp[x]==out[x] for x in [0,1]):
                type_2_1.append(t)
            else:
                if (out[0][0]*out[0][1])<(inp[0][0]*inp[0][1]):     
                    type_2_2.append(t)
                else:
                    type_2_3.append(t)
            
    return {'t1':type_1,'t1_1':type_1_1,'t1_2':type_1_2,'t1_3':type_1_3,'t1_4':type_1_4,
            't2': type_2, 't2_1':type_2_1,'t2_2':type_2_2,'t2_3':type_2_3}


def dimension_explained(dataset='train', plot='no'):
    print('------------',dataset, ' shapes ------------')
    diz=check_dim(dataset)
    print('t1 all inputs are equal: ', len(diz['t1']), 'of which: ')
    print('t1_1 also output equal: ', len(diz['t1_1']))
    print('t1_2 output smaller but fixed: ', len(diz['t1_2']))
    print('t1_3 output bigger but fixed: ', len(diz['t1_3']))
    print('t1_4 output size depends on input', len(diz['t1_4']))
    

    print('t2 input different: ', len(diz['t2']), 'of which: ')
    print('t2_1 output equal to input: ', len(diz['t2_1']))
    print('t2_2 output smaller : ', len(diz['t2_2']))
    print('t2_3 output bigger : ', len(diz['t2_3']))
    