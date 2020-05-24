"""
infinite generators of tasks for better training
"""

import numpy as np

def gener_one():
    """
    TASK 1 - Same dimension: Mirror right and down
    """
    
    skel = np.random.randint(2,size=(3,4))
    color = np.random.randint(1, 10)
    inp = np.where(skel==0, 0, color)
    flip_right = np.concatenate((inp,np.flip(inp, axis=1)), axis=1)
    out = np.concatenate((flip_right,np.flip(flip_right, axis=0)), axis=0)
    
    return inp, out 

def gener_two():
    """
    TASK 2 - Same dimension: Concatenate and flip central row
    """
    
    inp = np.random.randint(1,10,size=(2,2))
    flip = np.flip(inp, axis=1)
    conc1 = np.concatenate((inp, inp, inp), axis=1)
    conc2 = np.concatenate((flip, flip, flip), axis=1)
    out = np.concatenate((conc1, conc2, conc1), axis=0)
    
    return inp, out


def gener_three():
    """
    TASK 3 - Same dimension: Pattern filler
    """
    
    a = np.random.randint(2, 10, (8, 8))
    a = np.where(a==4, 1, a)
    m = np.tril(a) + np.tril(a, -1).T
    m = np.concatenate((m, np.flip(m, axis=0)), axis=0)
    out= np.concatenate((m, np.flip(m)), axis=1)
    inp = out.copy()

    p1 = np.random.randint(1, 13)
    p2 = np.random.randint(p1+2, p1+5)
    p3 = np.random.randint(1, 13)
    p4 = np.random.randint(p3+2, p3+5)

    inp[p1:p2, p3:p4] = 4

    p1 = np.random.randint(1, 13)
    p2 = np.random.randint(p1+2, p1+5)
    p3 = np.random.randint(1, 13)
    p4 = np.random.randint(p3+2, p3+5)

    inp[p1:p2, p3:p4] = 4 
    
    return inp, out

def gener_four():
    """
    TASK 4 - Different dimension: Denoise
    """
    
    dim1 = np.random.randint(10,15)
    dim2 = np.random.randint(13,18)
    out = np.zeros((dim1,dim2))
    col = np.random.choice([2,3,5])
    colno = np.random.choice([1,4,8])
    nsquare = np.random.randint(3,5)
    nnoise = np.random.randint(10,20)
    for sq in range(0,nsquare):
        p1 = np.random.randint(0, dim1-1)
        p2 = np.random.randint(p1+2, p1+8)
        p3 = np.random.randint(0, dim2-1)
        p4 = np.random.randint(p3+2, p3+8)

        out[p1:p2, p3:p4] = col

    inp = out.copy()

    for noise in range(0,nnoise):
        p1 = np.random.randint(0, dim1)
        p2 = np.random.randint(0, dim2)

        inp[p1,p2]=colno
        
    return inp, out

def gener_five():
    """
    TASK 5 - Different dimension: 2 squares
    """
    
    dim1 = np.random.randint(5,11)
    dim2 = np.random.randint(5,11)
    out = np.zeros((dim1,dim2))
    inp = out.copy()
    col = np.random.randint(1,10)
    col2 = np.random.randint(1,10)
    nsquare = np.random.randint(1,3)
    
    flippoints = np.random.randint(1,3)
    
    while col2 == col:
        col2 = np.random.randint(1,10)

    if nsquare == 1:
        p1 = np.random.randint(0, dim1-2)
        p2 = np.random.randint(p1+2,dim1)
        p3 = np.random.randint(0, dim2-2)
        p4 = np.random.randint(p3+2,dim2)

        out[p1:p2+1, p3:p4+1] = col
        inp[p1, p3] = col
        inp[p2, p4] = col
    if nsquare == 2:
        if dim1 > dim2:
            p1 = np.random.randint(0, dim1-3)
            p2 = np.random.randint(p1+1, dim1-2)
            p11 = np.random.randint(p2+1, dim1-1)
            p22 = np.random.randint(p11+1, dim1)
            p3 = np.random.randint(0, dim2-3)
            p4 = np.random.randint(p3+1, dim2)
            p33 = np.random.randint(0, dim2-3)
            p44 = np.random.randint(p33+1, dim2)

            out[p1:p2+1, p3:p4+1] = col            
            out[p11:p22+1, p33:p44+1] = col2
            
            if flippoints == 1:
                inp[p1, p3] = col
                inp[p2, p4] = col            
                inp[p22, p33] = col2
                inp[p11, p44] = col2
            else:
                inp[p2, p3] = col
                inp[p1, p4] = col            
                inp[p11, p33] = col2
                inp[p22, p44] = col2
                
        else:
            p1 = np.random.randint(0, dim1-3)
            p2 = np.random.randint(p1+1, dim1)
            p11 = np.random.randint(0, dim1-3)
            p22 = np.random.randint(p11+1, dim1)
            p3 = np.random.randint(0, dim2-3)
            p4 = np.random.randint(p3+1, dim2-2)
            p33 = np.random.randint(p4+1, dim2-1)
            p44 = np.random.randint(p33+1, dim2)

            out[p1:p2+1, p3:p4+1] = col            
            out[p11:p22+1, p33:p44+1] = col2
            
            if flippoints == 1:
                inp[p1, p3] = col
                inp[p2, p4] = col            
                inp[p22, p33] = col2
                inp[p11, p44] = col2
            else:
                inp[p2, p3] = col
                inp[p1, p4] = col            
                inp[p11, p33] = col2
                inp[p22, p44] = col2
                
    return inp, out

def gener_six():
    """
    TASK 6 - Different dimension: smallest squares
    """
    
    dim1 = np.random.randint(8,21)
    dim2 = np.random.randint(8,21)
    inp = np.zeros((dim1,dim2))
    nsquare = np.random.randint(2,6)
    colors = np.random.choice(range(1,10),5, replace=False)
    sqs=[]
    for i in range(0, nsquare):
        p1 = np.random.randint(1, dim1-3)
        p2 = np.random.randint(p1+1,dim1-1)
        p3 = np.random.randint(1, dim2-3)
        p4 = np.random.randint(p3+1,dim2-1)
        sqs.append([((p2-p1)*(p4-p3)), p1,p2,p3,p4, colors[i]])
        
    sortt = sorted(sqs, reverse=True)
    for sq in sortt:
        _,p1,p2,p3,p4,col = sq
        inp[p1:p2+1, p3:p4+1] = col
    _,p1,p2,p3,p4,col = sortt[-1]
    
    out = np.ones((p2-p1,p4-p3))*col
    
    return inp, out

def gener_seven():
    """
    TASK 7 - Different dimension: inverter
    """
    
    dim1 = np.random.randint(18,26)
    dim2 = np.random.randint(18,26)
    colors = np.random.choice(range(1,10),2, replace=False)
    inp = np.zeros((dim1,dim2))
    
    nnoise = np.random.randint(15,50)
    for noise in range(0,nnoise):
        p1 = np.random.randint(0, dim1)
        p2 = np.random.randint(0, dim2)

        inp[p1,p2] = colors[0]
        
    
    
    skel = np.random.randint(2,size=(3,3))
    out = skel*colors[0]
    topaste = np.kron(skel, np.ones((4,4)))*colors[1]
    
    p1 = np.random.randint(0, dim1-12)
    p2 = np.random.randint(0,dim2-12)
    
    inp[p1:p1+12, p2:p2+12] = topaste
    
    return inp, out


def task_builder(generator, n_train=1000, n_test=250):
    """
    creates the task generator
    
    :generator: generator for the wanted task
    :n_train: number of samples to generate for training
    :n_test: number of samples to generate for testing
    
    :returns: dictionary of the task of generated samples
    """
    
    task = {'train':[], 'test':[]}
    for i in range(0,n_train):
        inp, out = generator()
        task['train'].append({'input':inp, 'output':out})
    for i in range(0,n_test):
        inp, out = generator()
        task['test'].append({'input':inp, 'output':out})
        
    return task