# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 13:47:49 2024

@author: Hassan
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
import os
import math
import random
import copy
import gc

start_time=time.time()

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

# Example usage:
seed_value = 42
set_seed(seed_value)

torch.set_printoptions(precision=20, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=None)

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
warnings.filterwarnings("ignore")

def dy_dx(y, x):
  return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True, allow_unused=True)[0]
x = torch.tensor([[0.01*k for k in range(1,501)]],dtype=torch.float,requires_grad=True)
xx=torch.reshape(x,(1,500))

class mymodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.T1=nn.RNN(input_size=1, hidden_size=16,num_layers=3, batch_first=False)
        self.fc = nn.Linear(16, 1)

    def forward(self,x):
        temp1, h=self.T1(x)
        temp1 = self.fc(temp1)

        return temp1, h

model=mymodel()

def lossODE(y,a,b):
    mysum=torch.tensor([0.0],dtype=torch.float,requires_grad=True)
    xxx=x.squeeze()
    i=1
    mysum=(((y[i+1]-2*y[i]+y[i-1])/0.0001)+y[i]**0+(2.0/xxx[a])*((y[i+1]-y[i])/0.01))**2
    return mysum


def lossInitial():
    x0=torch.tensor([[0.0]],dtype=torch.float,requires_grad=True)
    x1=torch.tensor([[0.01]],dtype=torch.float,requires_grad=True)
    mysum1=(model(x0)[0]-1)**2
    mysum2=((model(x1)[0]-model(x0)[0])/0.01)**2
    mysum=mysum1+mysum2
    return mysum

def loss_fn(y,a,b):
    loss=lossODE(y,a,b)+lossInitial()
    optimizer.zero_grad()
    loss.backward(create_graph=False)
    optimizer.step()
    return loss.item()

optimizer = torch.optim.Adam(model.parameters(), lr=.001)

start_time = time.time()
losses=[]

loss=0
ten_mean_loss=10
vector_ten_losses=[10]*10
counter_for_ten_mean_loss=0
my_flag=False
for epoch in range(1000):
    sum_loss=0
    for zz in range(0,498):
        a=zz
        b=zz+2
        y=torch.zeros(3,dtype=torch.float)
        for i,value in enumerate(xx[0][a:b+1]):
            value=value.reshape((1,1))
            y[i]=model(value)[0].squeeze()
        
        
        loss=loss_fn(y,a,b)
        sum_loss += loss
        index=counter_for_ten_mean_loss % 10
        vector_ten_losses[index]=loss
        if counter_for_ten_mean_loss==10:
            counter_for_ten_mean_loss=0
        counter_for_ten_mean_loss+=1
        ten_mean_loss=np.mean(vector_ten_losses)
        if ten_mean_loss < 0.000001:
            my_flag=True
            break
    if my_flag==True:
        break

    if epoch % 10 ==0:
        print('epoch=',epoch,'  ,Loss=',sum_loss/100.0)

y_num=torch.zeros(500,dtype=torch.float)
residual=torch.zeros(500,dtype=torch.float)
for i,value in enumerate(xx[0]):
  value=value.reshape((1,1))
  y_num[i]=model(value)[0].squeeze()

plt.plot([i.item() for i in y_num])
plt.show()

temp=y_num.detach().numpy()
global zzz
zzz=torch.from_numpy(temp)
zzz.requires_grad_()


# Recurrent Neural Network Ended

#################################################################

# Monte Carlo Tree Search Starts


global max_allowable
max_allowable=30.0
global final_flag
final_flag=False

device = "cuda" if torch.cuda.is_available() else "cpu"


class MyModule(nn.Module):
    def __init__(self,vector):
        super().__init__()
        self.s, weightCounter=self.vec2exp(vector)
        if weightCounter==0:
          weightCounter=1
        w=0.01*torch.rand(weightCounter,dtype=torch.float,requires_grad=True)
        self.w = nn.Parameter(w) 
        
    def vec2exp(self,vector):
        vectorMaxLength=30
        s='@'
        weightCounter=0   
        for j,i in enumerate(vector):
                
            notFount=True        
            for counter,character in enumerate(s):
                
                if character =='@':
                    s_left=s[0:counter]
                    s_right=s[counter+1:]
                    myMod= i % 4
                    if (myMod==0 or myMod==1) and j > vectorMaxLength:
                        myMod=3
                    match myMod:
                        case 0:
                            s=s_left+'@$@'+s_right
                        case 1:
                            s=s_left+'(@$@)'+s_right
                        case 2:
                            s=s_left+'%(@)'+s_right
                        case 3:
                            s=s_left+'&'+s_right
                        case _:
                            pass
                    notFount=False
                    
                if character =='$':
                    s_left=s[0:counter]
                    s_right=s[counter+1:]
                    myMod= i % 4
                    match myMod:
                        case 0:
                            s=s_left+'+'+s_right
                        case 1:
                            s=s_left+'-'+s_right
                        case 2:
                            s=s_left+'/'+s_right
                        case 3:
                            s=s_left+'*'+s_right
                        # case 4:
                        #     s=s_left+'**'+s_right
                        case _:
                            pass
                    notFount=False
                    
                if character =='%':
                     s_left=s[0:counter]
                     s_right=s[counter+1:]
                     myMod= i % 4
                     match myMod:
                        case 0:
                            s=s_left+'torch.sin'+s_right
                        case 1:
                            s=s_left+'torch.cos'+s_right
                        case 2:
                            s=s_left+'torch.exp'+s_right
                        case 3:
                            s=s_left+'torch.log'+s_right
                     notFount=False
                    
                if character =='&':
                    s_left=s[0:counter]
                    s_right=s[counter+1:]
                    myMod= i % 2

                    if  myMod==0:
                        s=s_left+'x'+s_right
                    if myMod==1:
                        s=s_left+'self.w['+str(weightCounter)+']'+s_right
                        weightCounter =weightCounter+1                        
                    notFount=False
                    
                if notFount==False:
                    break;
                    
        return s,weightCounter
    def forward(self,x):
        return eval(self.s)

def vec2exp(vector):
    vectorMaxLength=30
    s='@'
    weightCounter=0   
    for j,i in enumerate(vector):
            
        notFount=True        
        for counter,character in enumerate(s):
            
            if character =='@':
                s_left=s[0:counter]
                s_right=s[counter+1:]
                myMod= i % 4
                if (myMod==0 or myMod==1) and j > vectorMaxLength:
    
                    myMod=3
                match myMod:
                    case 0:
                        s=s_left+'@$@'+s_right
                    case 1:
                        s=s_left+'(@$@)'+s_right
                    case 2:
                        s=s_left+'%(@)'+s_right
                    case 3:
                        s=s_left+'&'+s_right
                    case _:
                        pass
                notFount=False
                
            if character =='$':
                s_left=s[0:counter]
                s_right=s[counter+1:]
                myMod= i % 4
                match myMod:
                    case 0:
                        s=s_left+'+'+s_right
                    case 1:
                        s=s_left+'-'+s_right
                    case 2:
                        s=s_left+'/'+s_right
                    case 3:
                        s=s_left+'*'+s_right
                    case _:
                        pass
                notFount=False
                
            if character =='%':
                 s_left=s[0:counter]
                 s_right=s[counter+1:]
                 myMod= i % 4
                 match myMod:
                    case 0:
                        s=s_left+'torch.sin'+s_right
                    case 1:
                        s=s_left+'torch.cos'+s_right
                    case 2:
                        s=s_left+'torch.exp'+s_right
                    case 3:
                        s=s_left+'torch.log'+s_right
                 notFount=False
                
            if character =='&':
                s_left=s[0:counter]
                s_right=s[counter+1:]
                myMod= i % 2
                
                if  myMod==0:
                    s=s_left+'x'+s_right
                if myMod==1:
                    s=s_left+'self.w['+str(weightCounter)+']'+s_right
                    weightCounter =weightCounter+1
                      
                notFount=False
                
            if notFount==False:
                break;
    return s
############################################
############################################
############################################


class Node:
    def __init__(self,state,parent):
        self.state=state
        self.parent=parent
        self.children={}
        self.value=0
        self.n=0
        self.is_leaf=True
        self.num_of_actions=evaluate_num_of_actions(state)
        self.Termination,temp1_value,temp2_params=is_termination(state)
        self.params=[]
        if self.Termination==True:
            self.value=temp1_value
            self.params=temp2_params
        
def calculate_ucb(Node,N):
    C=1.0
    if Node.n==0:
        return float('inf')
    else:
        return float(Node.value)/Node.n+C*np.sqrt(np.log(N)/Node.n)

def expand(current_node):
    if current_node.Termination==True:
        return current_node
    else:
        current_node.is_leaf=False
        current_state=current_node.state
        for i in range(current_node.num_of_actions):
            temp=list([i])
            newState=current_state+temp
            new_node=Node(newState,current_node)
            current_node.children[i]=new_node
        return current_node.children[0]

def rolloout(current_node):
    ttt=copy.deepcopy(current_node)
    mysum=0
    cc=1
    ll1=len(ttt.state)
    ll2=int((1-ll1/max_allowable)*100)+10
    for _ in range(ll2):
        temp=copy.deepcopy(ttt)
        current_state=temp.state
        s=vec2exp(current_state)
        counter=len(current_state)
        while ('@' in s) or ('$' in s) or ('%' in s) or ('&' in s):
            myRand=np.random.randint(4)
            current_state.append(myRand)
            s=vec2exp(current_state)
            counter+=1
            if counter==max_allowable:
                break
        if ('@' in s) or ('$' in s) or ('%' in s) or ('&' in s):
            mysum+= 0.0
            continue
        x = torch.linspace(0.01, 5, 500, requires_grad=True)

        model=MyModule(current_state).to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001) 
        
        try:
            for i in range(1000):
                y_model = model(x)
                loss = loss_fn(y_model,zzz)
                optimizer.zero_grad(set_to_none=True)
                loss.backward(retain_graph=False)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            pars=model.w.clone().detach()
            cc+=1
            if math.isnan(loss.detach().numpy()):
                mysum+= 0.0
                continue
            else:
                if loss.item() < 1:
                    print('*****************************************************\n',file=open('MCTS_Results.txt','a'))
                    print('\n Expression=', model.s.replace('torch.','').replace('self.','').replace('[','(').replace(']',')'), file=open('MCTS_Results.txt','a'))
                    print('\n Parameters=',pars, file=open('MCTS_Results.txt','a'))
                    print('\n Loss=',loss.item(),file=open('MCTS_Results.txt','a'))
                    print('\n Main Expression in PyTorch=',model.s,file=open('MCTS_Results.txt','a'))
                    print('\n Branch=', current_state,file=open('MCTS_Results.txt','a'))
                #print('value=',1/(loss.item()+0.0000000000001))
                mysum+= 1/(loss.item()+0.0000000000001)
                if loss.item() < 0.00001:
                    global final_flag
                    final_flag=True
                    break
                continue
       
        except:
            pars=model.w.clone().detach()
            mysum+= 0.0
            continue
    return mysum/cc
    
def backpropogate(value,current_node):
    current_node.value += value
    current_node.n +=1
    current_node= current_node.parent
    while current_node != None:
        current_node.value += value
        current_node.n +=1
        current_node= current_node.parent
        
        
def select(current_node,N):
    while current_node.is_leaf !=True:
        length=len(current_node.children)
        list_of_ucb_values=[]
        for i in range(length):
            list_of_ucb_values.append(calculate_ucb(current_node.children[i],N))
        max_index=np.argmax(list_of_ucb_values)
        current_node=current_node.children[max_index]
    return current_node

def final_select(current_node,N):
    l=[]
    for i in range(4):
        l.append(current_node.children[i].value)
    max_index=np.argmax(l)
    current_node=current_node.children[max_index]
    return current_node
        
def evaluate_num_of_actions(state):
    if len(state)==0:
        return 4
    else:
        s=vec2exp(state)
        if ('@' in s) or ('$' in s) or ('%' in s) or ('&' in s):
            i=0
            while s[i] != '@' and s[i] != '$' and s[i] != '&' and s[i] != '%':
                i=i+1
            character=s[i]
            match character:
                case '@':
                    return 4
                case '$':
                    return 4
                case '%':
                    return 4
                case '&':
                    return 2
        else:
            return 0
def get_root(current_node):
    while current_node.parent !=None:
        current_node=current_node.parent
    return current_node
        
def is_termination(state):
    global max_allowable
    if len(state) > max_allowable:
        return True, -10.0, []
    else:
        s=vec2exp(state)
        if ('@' in s) or ('$' in s) or ('%' in s) or ('&' in s):
            return False, 0.0, []
        else:
            
            x = torch.linspace(0.01, 5, 500, requires_grad=True)

            Terminate=False
            model=MyModule(state).to(device)
            loss_fn = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=.001) 
            
            try:
                
                for i in range(1000):
                    y_model = model(x)
                    loss=loss_fn(y_model,zzz)
                    ################################
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward(retain_graph=False)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                pars=model.w.clone().detach()
                if math.isnan(loss.detach().numpy()):
                    return True,-10.0,pars
                else:
                    if loss.item() < 1:
                        print('*****************************************************\n',file=open('MCTS_Results.txt','a'))
                        print('\n Expression=', model.s.replace('torch.','').replace('self.','').replace('[','(').replace(']',')'), file=open('MCTS_Results.txt','a'))
                        print('\n Parameters=',pars, file=open('MCTS_Results.txt','a'))
                        print('\n Loss=',loss.item(),file=open('MCTS_Results.txt','a'))
                        print('\n Main Expression in PyTorch=',model.s,file=open('MCTS_Results.txt','a'))
                        print('\n Branch=', state,file=open('MCTS_Results.txt','a'))
                        if loss.item() < 0.00001:
                            global final_flag
                            final_flag=True
                    #print('value=',1/(loss.item()+0.0000000000001))
                    return True,1/(loss.item()+0.0000000000001), pars
            
            except:
                pars=model.w.clone().detach()
                return True,-10.0, pars
            
n_iterations=1000
N=0

root=Node([],None)
N1=Node([0],root)
N2=Node([1],root)
N3=Node([2],root)
N4=Node([3],root)
root.children[0]=N1
root.children[1]=N2
root.children[2]=N3
root.children[3]=N4
root.n=1
root.is_leaf=False

j=1
current_node=root
main_state=[]
flag=False
for kk in range(30):
    if flag==True:
        break
    if final_flag==True:
        break
    for iterations in range(20):
        if final_flag==True:
            break
        # if iterations % 10 ==0:
        #     print('Iteration=',iterations)
        N +=1
        current_node=select(current_node,N)
        if current_node.Termination ==True:
            j+=1
            if current_node.n==0:
                backpropogate(current_node.value, current_node)
            else:
                backpropogate(current_node.value, current_node)
        else:
            if current_node.n==0:
                value=rolloout(current_node)
                backpropogate(value, current_node)
            else:
                current_node=expand(current_node)
                value=rolloout(current_node)
                backpropogate(value, current_node)
        current_node=get_root(current_node)
    print('\n ##########Iteration=', kk)
    ch1=current_node.children[0]
    ch2=current_node.children[1]
    ch3=current_node.children[2]
    ch4=current_node.children[3]
    pref=final_select(current_node,N)
    main_state=pref.state
    del current_node, ch1, ch2, ch3, ch4
    gc.collect()
    N=0

    root=Node(main_state,None)
    N1=Node(main_state+[0],root)
    N2=Node(main_state+[1],root)
    N3=Node(main_state+[2],root)
    N4=Node(main_state+[3],root)
    root.children[0]=N1
    root.children[1]=N2
    root.children[2]=N3
    root.children[3]=N4
    root.n=1
    root.is_leaf=False
    j=1
    current_node=root
    ss=vec2exp(root.state)
    if not(('@' in ss) or ('$' in ss) or ('%' in ss) or ('&' in ss)) :
        flag=True
        break

end_time=time.time()

execution_time=end_time-start_time

print(f"Execution time: {execution_time} seconds",file=open('exacution_time.txt','a'))