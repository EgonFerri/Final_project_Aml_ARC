import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import torch.nn.functional as F


def get_task(data_path='data', subset='train', index=0, print_path=False):
    """
    gets a task for the needed subset

    :subset: subset of the task train/eval/test
    :index: parameter for returning the tasks of a given index
    :print_path: if set = True prints the path of the subtask

    :returns: the dictionary of the subtask
    """

    data_path = Path(data_path)
    training_path = data_path / 'training'
    evaluation_path = data_path / 'evaluation'
    test_path = data_path / 'test'

    training_tasks = sorted(os.listdir(training_path))
    evaluation_tasks = sorted(os.listdir(evaluation_path))
    test_tasks = sorted(os.listdir(test_path))

    if subset == 'train':
        task_file = str(training_path / training_tasks[index])
    elif subset == 'eval':
        task_file = str(evaluation_path / evaluation_tasks[index])
    else:
        task_file = str(test_path / test_tasks[index])
    with open(task_file, 'r') as f:
        task = json.load(f)
    if print_path == True:
        print(task_file)

    return task



def plot_task(task):
    """
    plots the training samples for train and test tasks

    :task: task which is wanted to plot
    """

    cmap = colors.ListedColormap(
    ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)

    n = len(task["train"]) + len(task["test"])
    fig, axs = plt.subplots(2, n, figsize=(4*n,8), dpi=50)
    plt.subplots_adjust(wspace=0, hspace=0)
    fig_num = 0

    for i, t in enumerate(task["train"]):
        t_in, t_out = np.array(t["input"]), np.array(t["output"])
        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)
        axs[0][fig_num].set_title(f'Train-{i} in')
        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)
        axs[1][fig_num].set_title(f'Train-{i} out')
        fig_num += 1

    for i, t in enumerate(task["test"]):
        t_in, t_out = np.array(t["input"]), np.array(t.get('output', [[0,0],[0,0]]))
        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)
        axs[0][fig_num].set_title(f'Test-{i} in')
        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)
        axs[1][fig_num].set_title(f'Test-{i} out')
        fig_num += 1

    plt.tight_layout()
    plt.show()


def check_dim(data_path, dataset='train'):
    """
    checks the dimensions of the tasks

    :dataset: type of dataset to check the dimension of

    :returns: dictionary of the tasks and the length of the different subtypes of tasks
    """

    type_1 = []
    type_1_1 = []
    type_1_2 = []
    type_1_3 = []
    type_1_4 = []

    type_2 = []
    type_2_1 = []
    type_2_2 = []
    type_2_3 = []

    leng = 400
    if dataset == 'test':
        leng=100
    for t in range(0, leng):
        task = get_task(data_path, dataset, t)
        inp = []
        out = []
        for i in task['train']:
            inp.append(np.array(i['input']).shape)
            out.append(np.array(i['output']).shape)
        if all(x == inp[0] for x in inp):
            type_1.append(t)
            if all(x == inp[0] for x in out):
                type_1_1.append(t)
            else:
                if all(x == out[0] for x in out):
                    if (out[0][0]*out[0][1])<(inp[0][0]*inp[0][1]):
                        type_1_2.append(t)
                    else:
                        type_1_3.append(t)
                else:
                    type_1_4.append(t)
        else:
            type_2.append(t)
            if all(inp[x] == out[x] for x in [0,1]):
                type_2_1.append(t)
            else:
                if (out[0][0]*out[0][1])<(inp[0][0]*inp[0][1]):
                    type_2_2.append(t)
                else:
                    type_2_3.append(t)

    return {'t1':type_1,'t1_1':type_1_1,'t1_2':type_1_2,'t1_3':type_1_3,'t1_4':type_1_4,
            't2':type_2, 't2_1':type_2_1,'t2_2':type_2_2,'t2_3':type_2_3}


def dimension_explained(data_path, dataset = 'train'):
    """
    prints the various types of tasks and their caracteristics

    :dataset: type of dataset to show the caracteristics of
    """

    print('------------', dataset, ' shapes ------------')
    dic = check_dim(data_path, dataset)
    print('t1 all inputs are equal: ', len(dic['t1']), 'of which: ')
    print('t1_1 also output equal: ', len(dic['t1_1']))
    print('t1_2 output smaller but fixed: ', len(dic['t1_2']))
    print('t1_3 output bigger but fixed: ', len(dic['t1_3']))
    print('t1_4 output size depends on input: ', len(dic['t1_4']))


    print('t2 input different: ', len(dic['t2']), 'of which: ')
    print('t2_1 output equal to input: ', len(dic['t2_1']))
    print('t2_2 output smaller: ', len(dic['t2_2']))
    print('t2_3 output bigger: ', len(dic['t2_3']))


def pad_crop(x, desired_w, desired_h, current_w, current_h, goal):
    """
    pads or crops array into desired shape

    :x: array to reshape
    :desired_w: desired width
    :desired_h: desired height
    :current_w: width of array to reshape
    :current_h: height of array to reshape
    :goal: if set to 'pad' pads the array, if set to 'crop' crops the array

    :returns: padded or cropped array
    """

    diff_w = np.abs(desired_w-current_w)
    if diff_w % 2 == 0:
        if goal == 'pad':
            x = F.pad(x, (0, 0, int(diff_w/2), int(diff_w/2)), mode='constant', value=0)
        elif goal == 'crop':
            if -int(diff_w/2) == 0:
                x = x[:, :, int(diff_w/2):, :]
            else:
                x = x[:, :, int(diff_w/2):-int(diff_w/2), :]
        else:
            pass
    else:
        if goal == 'pad':
            x = F.pad(x, (0, 0, int(diff_w/2)+1, int(diff_w/2)), mode='constant', value=0)
        elif goal == 'crop':
            if -int(diff_w/2) == 0:
                x = x[:, :, int(diff_w/2)+1:, :]
            else:
                x = x[:, :, int(diff_w/2)+1:-int(diff_w/2), :]
        else:
            pass

    diff_h = np.abs(desired_h-current_h)
    if diff_h % 2 == 0:
        if goal == 'pad':
            x = F.pad(x, (int(diff_h/2), int(diff_h/2), 0, 0), mode='constant', value=0)
        elif goal == 'crop':
            if (-int(diff_h/2)) == 0:
                x = x[:, :, :, int(diff_h/2):]
            else:
                x = x[:, :, :, int(diff_h/2):-int(diff_h/2)]
        else:
            pass
    else:
        if goal == 'pad':
            x = F.pad(x, (int(diff_h/2)+1, int(diff_h/2), 0, 0), mode='constant', value=0)
        elif goal == 'crop':
            if (-int(diff_h/2)) == 0:
                x = x[:, :, :, int(diff_h/2)+1:]
            else:
                x = x[:, :, :, int(diff_h/2)+1:-int(diff_h/2)]
        else:
            pass

    return x

def expand(x):
    """
    expands an array to 10 channels (one for each color)

    :x: array to expand

    :returns: expanded array
    """

    img = np.array([np.zeros((x.shape[0], x.shape[1]))+i for i in range(10)])
    img = (x-img == 0)*1
    return img
