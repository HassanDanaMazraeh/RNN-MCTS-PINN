# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 12:37:48 2024

@author: Hassan
"""

import torch
import numpy as np
import warnings
import os
import random
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
warnings.filterwarnings("ignore") 

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

seed_value = 42
set_seed(seed_value)

torch.set_printoptions(precision=20, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=None)


def dy_dx(y, x):
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y), create_graph=True,allow_unused=True
    )[0]

x = torch.linspace(0.0, 5, 5001, requires_grad=True)


w=torch.tensor([-0.16138416528701782227, -0.98353004455566406250, -0.98389834165573120117],requires_grad=True,dtype=torch.float);



optimizer = torch.optim.Adam([w], lr=0.00001) 

losses=[]

for i in range(10000):
    y=((w[0]*(x*x))+w[1]*w[2])

    y_p = dy_dx(y, x)      
    y_pp = dy_dx(y_p, x)
    

    residual = x*y_pp+2.0*y_p+x*1.0
    initial1 = 100*(y[0]-1)**2
    initial2= 10*(y_p[0])**2
    loss = (residual**2).mean() + initial1+initial2
    
    losses.append(loss.item())

    optimizer.zero_grad(set_to_none=True)
    loss.backward(retain_graph=False)
    #nn.utils.clip_grad_value_(model.parameters(), 1)
    optimizer.step()
    del y_p
    del y_pp
    optimizer.zero_grad(set_to_none=True)
    if i % 10 ==0:
        print('Iteration=', i, 'loss=', loss.item())


print('parameters=',w)
