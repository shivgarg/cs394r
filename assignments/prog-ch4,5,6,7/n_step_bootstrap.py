from typing import Iterable, Tuple

import numpy as np
from env import EnvSpec
from policy import Policy

def on_policy_n_step_td(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    n:int,
    alpha:float,
    initV:np.array
) -> Tuple[np.array]:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        n: how many steps?
        alpha: learning rate
        initV: initial V values; np array shape of [nS]
    ret:
        V: $v_pi$ function; numpy array shape of [nS]
    """
    V = np.array(initV)

    for traj in trajs:
        T = len(traj)
        G = 0
        for i in range(T+n-1):
            tau = i-n+1
            if tau >=0:
                G = 0 
                for j in range(tau+1, min(tau+n,T)+1):
                    G += env_spec.gamma**(j-tau-1)*traj[j-1][2]
                if tau+n < T:
                    G += env_spec.gamma**n*V[traj[tau+n][0]]
                V[traj[tau][0]] += alpha*(G-V[traj[tau][0]])

    #####################
    # TODO: Implement On Policy n-Step TD algorithm
    # sampling (Hint: Sutton Book p. 144)
    #####################

    return V

def off_policy_n_step_sarsa(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    n:int,
    alpha:float,
    initQ:np.array
) -> Tuple[np.array,Policy]:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        n: how many steps?
        alpha: learning rate
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_star$ function; numpy array shape of [nS,nA]
        policy: $pi_star$; instance of policy class
    """
    Q = np.array(initQ)
    eps = 0.1
    pi = egreedy_policy(Q,eps)
    for traj in trajs:
        T = len(traj)
        for t in range(T+n-1):
            tau = t-n+1
            if tau >= 0:
                rho = 1
                G = 0
                for i in range(tau+1, min(tau+n,T-1)+1):
                    rho *= pi.action_prob(traj[i][0], traj[i][1])/bpi.action_prob(traj[i][0], traj[i][1])
                for i in range(tau+1, min(tau+n,T)+1):
                    G += env_spec.gamma**(i-tau-1)*traj[i-1][2]
                if tau+n < T:
                    G += env_spec.gamma**n*Q[traj[tau+n][0]][traj[tau+n][1]]
                Q[traj[tau][0]][traj[tau][1]] += alpha*rho*(G-Q[traj[tau][0]][traj[tau][1]])
                pi = egreedy_policy(Q,eps)


    #####################
    # TODO: Implement Off Policy n-Step SARSA algorithm
    # sampling (Hint: Sutton Book p. 149)
    #####################
    assignment = np.zeros(env_spec.nS)
    for i in range(env_spec.nS):
        assignment[i] = np.argmax(Q[i])
    return Q, optimal_policy(assignment)

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

class egreedy_policy(Policy):
    def __init__(self, Q:np.array, eps:float):
        self.Q = Q
        self.eps = eps

    def action_prob(self, state:int, action:int):
        if np.argmax(self.Q[state]) == action:
            return 1-self.eps + (self.eps/len(self.Q[state]))
        else:
            return self.eps/len(self.Q[state])
    
    def action(self, state:int):
        if np.random.random() > self.eps:
            return np.argmax(self.Q[state])
        else:
            return np.random.randint(0,len(self.Q[state]))