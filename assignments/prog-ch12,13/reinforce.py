from typing import Iterable
import numpy as np
import torch
import torch.nn as nn


class NN(nn.Module):
    def __init__(self,state_dim, num_output):
        super(NN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32,num_output)
        )
    
    def forward(self,s):
        s = torch.Tensor(s)
        return self.network(s)

class PiApproximationWithNN():
    def __init__(self,
                 state_dims,
                 num_actions,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        action_dims: the number of possible actions
        alpha: learning rate
        """
        
        # Tips for TF users: You will need a function that collects the probability of action taken
        # actions; i.e. you need something like
        #
            # pi(.|s_t) = tf.constant([[.3,.6,.1], [.4,.4,.2]])
            # a_t = tf.constant([1, 2])
            # pi(a_t|s_t) =  [.6,.2]
        #
        # To implement this, you need a tf.gather_nd operation. You can use implement this by,
        #
            # tf.gather_nd(pi,tf.stack([tf.range(tf.shape(a_t)[0]),a_t],axis=1)),
        # assuming len(pi) == len(a_t) == batch_size
        self.network = NN(state_dims,num_actions)
        self.optim = torch.optim.Adam(self.network.parameters(), lr=alpha)
        
    def __call__(self,s) -> int:
        logits = self.network(s)
        dist = torch.distributions.categorical.Categorical(logits=logits)
        return dist.sample().item()

    def update(self, s, a, gamma_t, delta):
        """
        s: state S_t
        a: action A_t
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """
        self.optim.zero_grad()
        logits = self.network(s).unsqueeze(0)
        loss = gamma_t* delta* torch.nn.functional.cross_entropy(logits, torch.tensor([a]))
        loss.backward()
        self.optim.step()

class Baseline(object):
    """
    The dumbest baseline; a constant for every state
    """
    def __init__(self,b):
        self.b = b

    def __call__(self,s) -> float:
        return self.b

    def update(self,s,G):
        pass

class VApproximationWithNN(Baseline):
    def __init__(self,
                 state_dims,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        alpha: learning rate
        """
        self.network = NN(state_dims,1)
        self.optim = torch.optim.Adam(self.network.parameters(), lr=alpha)
        
    def __call__(self,s) -> float:
        return self.network(s).detach().item()

    def update(self,s,G):
        self.optim.zero_grad()
        val = self.network(s)
        loss = 0.5* torch.nn.functional.mse_loss(val, torch.tensor([G]))
        loss.backward()
        self.optim.step()


def REINFORCE(
    env, #open-ai environment
    gamma:float,
    num_episodes:int,
    pi:PiApproximationWithNN,
    V:Baseline) -> Iterable[float]:
    """
    implement REINFORCE algorithm with and without baseline.

    input:
        env: target environment; openai gym
        gamma: discount factor
        num_episode: #episodes to iterate
        pi: policy
        V: baseline
    output:
        a list that includes the G_0 for every episodes.
    """
    ans = []
    for i in range(num_episodes):
        states = []
        rewards = []
        actions = []
        done = False
        s = env.reset()
        states.append(s)
        G = 0
        t = 0
        while not done:
            a = pi(states[-1])
            s,r,done,_ = env.step(a)
            if not done:
                states.append(s)
            rewards.append(r)
            actions.append(a)
            G += gamma**t*r
            t+=1
        ans.append(G)
        g_t = 1
        for t in range(len(states)):
            G = (G - rewards[t])/gamma
            val = V(states[t])
            delta = G - val
            V.update(states[t],G)
            g_t *= gamma
            pi.update(states[t],actions[t],g_t, delta)
    return ans 