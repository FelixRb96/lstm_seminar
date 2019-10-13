"""
Lets implement an lstm from my head and see if I got the basics of torch
"""

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch import distributions
import numpy as np
import matplotlib.pyplot as plt

class LSTM(nn.Module):
    
    def __init__(self, input_features, output_features):
        super(LSTM, self).__init__()
        self.hidden_state = torch.zeros(output_features) 
        self.prediction = torch.zeros(output_features) 

        self.forget = nn.Linear(input_features + output_features, output_features)
        self.input = nn.Linear(input_features + output_features, output_features)
        self.block_input = nn.Linear(input_features + output_features, output_features)
        self.output = nn.Linear(input_features + output_features, output_features)

    def forget_gate(self, x):
        return torch.sigmoid(self.forget(x))

    def input_gate(self, x):
        inpt = torch.tanh(self.input(x))
        block_inpt = torch.sigmoid(self.input(x))

        return  inpt * block_inpt

    def output_gate(self, x):
        return torch.sigmoid(self.output(x))

    def forward(self, x):
        x = torch.cat((x, self.prediction)) 
        
        forget_gate = self.forget_gate(x)
        input_gate = self.input_gate(x)
        output_gate = self.output_gate(x)

        self.hidden_state = forget_gate * self.hidden_state
        self.hidden_state = input_gate + self.hidden_state
        self.prediction = output_gate * torch.relu(self.hidden_state)
        return self.prediction

class optimized_LSTM(torch.jit.ScriptModule):
    """
    TODO: optimize to death

    Taking in 6 parameters and outputting 6 parameters
    """
    __constants__ = ['hidden_state', 'prediction']

    def __init__(self):
        super(optimized_LSTM, self).__init__()
        self.hidden_state = torch.nn.Parameter(torch.zeros(6))
        self.prediction = torch.nn.Parameter(torch.zeros(6))
        self.forget = nn.Linear(12,6) 
        self.input = nn.Linear(12,6)
        self.block_input = nn.Linear(12,6)
        self.output = nn.Linear(12,6)
    
    @torch.jit.script_method
    def forward(self, x):
        x = torch.cat((x, self.prediction)) 
        
        forget_gate = torch.sigmoid(self.forget(x))
        
        inpt = torch.tanh(self.input(x))
        block_inpt = torch.sigmoid(self.input(x))


        input_gate = inpt * block_inpt
        output_gate = torch.sigmoid(self.output(x))

        self.hidden_state = forget_gate * self.hidden_state
        self.hidden_state = input_gate + self.hidden_state
        self.prediction = output_gate * torch.relu(self.hidden_state)
        return self.prediction
        
    

def train(model, x_train, y_train, x_test, y_test, err_thresh, max_epochs = 50, learning_rate = 0.01):
    """
    train a model with data and optimize the weights
    """
    opt = optim.SGD(model.parameters(), lr = learning_rate)
    epoch = 0
    loss_fct = nn.MSELoss()
    loss = err_thresh + 1
    
    while epoch < max_epochs and loss > err_thresh:
        for x,y in zip(x_train, y_train):
            
            x = x.reshape(1)
            opt.zero_grad()
            pred = model(x)
            loss = loss_fct(pred, y) 
            loss.backward(retain_graph = True)

            opt.step()
            print(i)

        print(f"epoch {epoch} Done\n")
        preds = torch.zeros(99)
        for i,x in enumerate(x_test):
            if i < 99:
                x= x.reshape(1)
                preds[i] = model(x)
            else:
                pass

        loss = F.mse_loss(preds, y_test)
        print(f"loss: {loss}\n")
        epoch += 1

def train_gen(model, x_train_gen, y_train_gen, x_test_gen, y_test_gen, err_thresh, max_epochs = 50, learning_rate = 0.01):
    """
    train a model with data and optimize the weights
    """
    opt = optim.SGD(model.parameters(), lr = learning_rate)
    epoch = 0
    loss_fct = nn.MSELoss()
    loss = err_thresh + 1
    
    while epoch < max_epochs and loss > err_thresh:
        for i, (x,y) in enumerate(zip(x_train_gen, y_train_gen)):
            
            opt.zero_grad()
            pred = model(x)
            loss = loss_fct(pred, y) 
            loss.backward(retain_graph = True)

            opt.step()
            if i % 1000 == 0:
                print(f"did {i} steps so far")
        
        print(f"epoch {epoch} Done\n")
        y_test = torch.Tensor(y_test)
        preds = torch.zeros(len(y_test))
        for x in x_test_gen:
            preds[i] = model(x)

        loss = F.mse_loss(preds, y_test)
        print(f"loss: {loss}\n")
        epoch += 1

def _test(phi = 0.6):
    model = LSTM(1,1)
    x = torch.randn(100)
    y = torch.randn(100)

    for i in range(1, len(x)):
        x[i] += phi * x[i-1] 
        y[i] += phi * y[i-1] 

    train(model, x[0:99], x[1:100], y[0:99], y[1:100], 1)
