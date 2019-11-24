import numpy as np
from algo import ValueFunctionWithApproximation
import math
import torch
import torch.nn as nn


class linear(nn.Module):
    def __init__(self, num_tiles, num_tilings):
        super(linear, self).__init__()
        self.fc = nn.Linear(num_tiles*num_tilings,1, bias=False)

    def forward(self, s):
        return self.fc(s)

class ValueFunctionWithTile(ValueFunctionWithApproximation):
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_tilings:int,
                 tile_width:np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maximum value for each dimension in state
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        self.state_low = state_low.flatten()
        self.state_high = state_high.flatten()
        self.num_tilings = num_tilings
        self.tile_width = tile_width.flatten()
        self.offsets = []
        num_dim = len(state_high)
        accum = 1
        for i in range(num_dim-1,-1,-1):
            self.offsets.append(accum)
            accum *= math.ceil((state_high[i]-state_low[i])/(tile_width[i])) + 1
        
        self.num_tiles = accum
        self.offsets.reverse()
        # Linear layer
        self.model = linear(self.num_tiles, self.num_tilings)

    def get_state_vector(self,s):
        state = np.zeros(self.num_tiles*self.num_tilings)
        s = s.flatten()
        num_dims = len(s)
        offset = 0
        for i in range(self.num_tilings):
            offset = i*self.num_tiles
            for j in range(num_dims):
                low = (self.state_low[j]-(i/self.num_tilings)*self.tile_width[j])
                val = s[j]
                tile_id = math.floor((val-low)/self.tile_width[j])

                offset += tile_id*self.offsets[j]
            state[offset] = 1
        return state

    def __call__(self,s):
        state = torch.Tensor(self.get_state_vector(s))
        return self.model(state).detach().item()

    def update(self,alpha,G,s_tau):
        optim = torch.optim.SGD(self.model.parameters(),lr=alpha)
        optim.zero_grad()
        state = torch.Tensor(self.get_state_vector(s_tau))
        prediction = self.model(state)
        G = torch.Tensor([G])
        loss = 0.5*nn.functional.mse_loss(prediction, G)
        loss.backward()
        optim.step()

        return None
