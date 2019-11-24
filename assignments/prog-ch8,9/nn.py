import numpy as np
from algo import ValueFunctionWithApproximation

#import tensorflow as tf
import torch
import torch.nn as nn

class NN(nn.Module):
    def __init__(self, state_dims):
        super(NN, self).__init__()
        self.hidden = nn.Sequential(
                        nn.Linear(state_dims, 32),
                        nn.ReLU(),
                        nn.Linear(32,32),
                        nn.ReLU(),
                        nn.Linear(32,1)
                    )
        
    def forward(self, x):
        x = torch.Tensor(x).detach()
        return self.hidden(x)

class ValueFunctionWithNN(ValueFunctionWithApproximation):
    def __init__(self,
                 state_dims):
        """
        state_dims: the number of dimensions of state space
        """
        self.model = NN(state_dims)
        self.optim = torch.optim.Adam(self.model.parameters(),lr=0.001,betas=(0.9,0.999))

    def __call__(self,s):
        return self.model(s).detach().item()

    def update(self,alpha,G,s_tau):
        self.optim.zero_grad()
        G = torch.Tensor([G])
        loss = 0.5*nn.functional.mse_loss(self.model(s_tau), G.detach())
        loss.backward()
        self.optim.step()
        return None
