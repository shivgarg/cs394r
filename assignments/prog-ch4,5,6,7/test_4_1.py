#
# @file test_4_1.py
# @author ZXJ, MJL, JQG @ UTAustin
# @date Sep. 18 2019
# @brief Testbench for GridWorld
#

import numpy as np
from tqdm import tqdm

from env import EnvSpec, Env, EnvWithModel
from policy import Policy

from dp import value_iteration, value_prediction
from monte_carlo import off_policy_mc_prediction_ordinary_importance_sampling as mc_ois
from monte_carlo import off_policy_mc_prediction_weighted_importance_sampling as mc_wis
from n_step_bootstrap import off_policy_n_step_sarsa as nsarsa
from n_step_bootstrap import on_policy_n_step_td as ntd


class RandomPolicy(Policy):
    def __init__(self, nA, p=None):
        self.p = p if p is not None else np.array([1/nA]*nA)

    def action_prob(self, state, action=None):
        return self.p[action]

    def action(self, state):
        return np.random.choice(len(self.p), p=self.p)


class GridWorld(EnvWithModel):  
    # GridWorld for example 4.1
    def __init__(self):
        # 16 states: 0 and 15 terminal
        # 4 action: 0 left, 1 up, 2 right, 3 down
        env_spec = EnvSpec(16, 4, 1.)
        super().__init__(env_spec)
        self.trans_mat, self.ret_mat = self._build_trans_mat()
        self.terminal_state = [0, 15]

    def _build_trans_mat(self):
        trans_mat = np.zeros((16, 4, 16), dtype=int)
        ret_mat = np.zeros((16, 4, 16)) - 1.

        for s in range(1, 15):
            if s % 4 == 0:
                trans_mat[s][0][s] = 1.
            else:
                trans_mat[s][0][s-1] = 1.
            if s < 4:
                trans_mat[s][1][s] = 1.
            else:
                trans_mat[s][1][s-4] = 1.
            if (s+1) % 4 == 0:
                trans_mat[s][2][s] = 1.
            else:
                trans_mat[s][2][s+1] = 1.
            if s > 11:
                trans_mat[s][3][s] = 1.
            else:
                trans_mat[s][3][s+4] = 1.

        for a in range(4):
            trans_mat[0][a][0] = 1.
            trans_mat[15][a][15] = 1.
            ret_mat[0][a][0] = 0
            ret_mat[15][a][15] = 0

        return trans_mat, ret_mat

    @property
    def TD(self):
        return self.trans_mat

    @property
    def R(self):
        return self.ret_mat

    def reset(self):
    # Random initialze location for each episode run
        self.state = np.random.randint(1, 15)
        return self.state

    def step(self, action):
        assert action in range(self.spec.nA), "Invalid Action"
        assert self.state not in self.terminal_state, "Episode has ended!"

        prev_state = self.state
        self.state = np.random.choice(self.spec.nS, p=self.trans_mat[self.state, action])
        r = self.ret_mat[prev_state, action, self.state]

        if self.state in self.terminal_state:
            return self.state, r, True
        else:
            return self.state, r, False


def visualize(pi):
    # Visulize policy with some strings
    visual_policy = np.empty(16, dtype=object)
    for s in range(16):
        if pi.action(s) == 0:
            visual_policy[s] = '<-' # left
        elif pi.action(s) == 1:
            visual_policy[s] = 'up' # up
        elif pi.action(s) == 2:
            visual_policy[s] = '->' # right
        elif pi.action(s) == 3:
            visual_policy[s] = 'dn' # down
    return visual_policy


def Q2V(Q, pi):
    # Compute V based on Q and policy pi
    V = np.zeros(16)
    for s in range(16):
        V[s] = 0
        for a in range(4):
            V[s] += pi.action_prob(s, a) * Q[s, a]
    return V


if __name__ == "__main__":

    grid_world = GridWorld()
    behavior_policy = RandomPolicy(4)
    initV = np.zeros(16)

    # Sample with random policy
    N_EPISODES = 10000

    print("Generating episodes based on random policy")
    trajs = []
    for _ in tqdm(range(N_EPISODES)):
        s = grid_world.reset()
        traj = []

        while s != 0 and s != 15:
            a = behavior_policy.action(s)
            next_s, r, _ = grid_world.step(a)
            traj.append((s, a, r, next_s))
            s = next_s
        trajs.append(traj)


    print("DP value prediction under random policy")
    V, Q = value_prediction(grid_world, behavior_policy, initV, 1e-12)
    print(V.reshape((4, 4)))

    print("DP value iteration optimal value and policy")
    V, pi = value_iteration(grid_world, initV, 1e-12)
    print(V.reshape((4, 4)))
    print(visualize(pi).reshape((4, 4)))

    # On-policy evaluation tests for random policy
    # OIS
    Q_est_ois = mc_ois(grid_world.spec, trajs, behavior_policy, behavior_policy,
                       np.zeros((grid_world.spec.nS, grid_world.spec.nA)))
    # WIS
    Q_est_wis = mc_wis(grid_world.spec, trajs, behavior_policy, behavior_policy,
                       np.zeros((grid_world.spec.nS, grid_world.spec.nA)))
    # 3-step TD with alpha = 0.005
    #V_est_td = ntd(grid_world.spec, trajs, 3, 0.005, np.zeros((grid_world.spec.nS)))

    print("On random policy value OIS: ")
    print(Q2V(Q_est_ois, behavior_policy).reshape((4, 4)))
    print("On random policy value WIS: ")
    print(Q2V(Q_est_wis, behavior_policy).reshape((4, 4)))
    print("3-step TD value estimation on random policy: ")
    #print(V_est_td.reshape((4, 4)))

    # Off-policy evaluation test with optimal policy
    Q_est_ois = mc_ois(grid_world.spec, trajs, behavior_policy, pi, np.zeros((grid_world.spec.nS, grid_world.spec.nA)))
    Q_est_wis = mc_wis(grid_world.spec, trajs, behavior_policy, pi, np.zeros((grid_world.spec.nS, grid_world.spec.nA)))
    print("Off policy evaluation for optimal value OIS: ")
    print(Q2V(Q_est_ois, pi).reshape((4, 4)))
    print("Off policy evaluation for optimal value WIS: ")
    print(Q2V(Q_est_wis, pi).reshape((4, 4)))

    # Off-policy SARSA
    # 3-step with alpha = 0.01, should converge to v*
    #Q_star_est, pi_star_est = nsarsa(grid_world.spec, trajs, behavior_policy, n=3, alpha=0.01,
                                     #initQ=np.zeros((grid_world.spec.nS, grid_world.spec.nA)))
    #print("3-step SARSA off policy optimal value est. :")
    #print(Q2V(Q_star_est, pi_star_est).reshape((4, 4)))
    #print("3-step SARSA off policy optimal policy :")
    #print(visualize(pi_star_est).reshape((4, 4)))
