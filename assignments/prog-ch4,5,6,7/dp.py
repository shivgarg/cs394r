from typing import Tuple

import numpy as np
from env import EnvWithModel
from policy import Policy

def value_prediction(env:EnvWithModel, pi:Policy, initV:np.array, theta:float) -> Tuple[np.array,np.array]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        pi: policy
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        V: $v_\pi$ function; numpy array shape of [nS]
        Q: $q_\pi$ function; numpy array shape of [nS,nA]
    """

    #####################
    # TODO: Implement Value Prediction Algorithm (Hint: Sutton Book p.75)
    #####################
    num_states = env.spec.nS
    num_actions = env.spec.nA
    V = np.array(initV)
    Q = np.zeros((num_states,num_actions))
    R = env.R
    TD = env.TD

    change = theta + 1
    while change > theta:
        change = 0 
        for i in range(num_states):
            old_v = V[i]
            new_v = 0
            for j in range(num_actions):
                sum_a = 0
                for k in range(num_states):
                    sum_a += TD[i,j,k]*(R[i,j,k]+env.spec.gamma*V[k])
                new_v += pi.action_prob(i,j)*sum_a
            change = max(change, abs(new_v-old_v))
            V[i] = new_v

    for i in range(num_states):
        for j in range(num_actions):
            for k in range(num_states):
                Q[i][j] += TD[i,j,k]*(R[i,j,k]+env.spec.gamma*V[k])     
    return V, Q

def value_iteration(env:EnvWithModel, initV:np.array, theta:float) -> Tuple[np.array,Policy]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        value: optimal value function; numpy array shape of [nS]
        policy: optimal deterministic policy; instance of Policy class
    """

    #####################
    # TODO: Implement Value Iteration Algorithm (Hint: Sutton Book p.83)
    #####################
    V = np.array(initV)
    num_states = env.spec.nS
    num_actions = env.spec.nA
    TD = env.TD
    R = env.R
    change = theta+1
    while change > theta:
        change = 0
        for i in range(num_states):
            old_v = V[i]
            new_v = np.NINF
            for j in range(num_actions):
                sum_a = 0
                for k in range(num_states):
                    sum_a += TD[i,j,k]*(R[i,j,k]+ env.spec.gamma*V[k])
                new_v = max(new_v, sum_a)
            change = max(change, abs(new_v-old_v))
            V[i] = new_v
    
    opt_action = np.zeros(num_states,dtype=int)
    for i in range(num_states):
        action_values  = np.zeros(num_actions)
        for j in range(num_actions):
            sum_a = 0
            for k in range(num_states):
                sum_a += TD[i,j,k]*(R[i,j,k]+ env.spec.gamma*V[k])
            action_values[j] = sum_a
        opt_action[i] = np.argmax(action_values)
    pi = optimal_policy(opt_action)

    return V, pi


class optimal_policy(Policy):
    def __init__(self, assignment):
        self.assignment = assignment

    def action_prob(self, state:int, action:int):
        if self.assignment[state] == action:
            return 1
        else:
            return 0
    
    def action(self, state:int):
        return self.assignment[state]