import numpy as np
import sys

NUM_ARMS = 10
RUN_LENGTH = 10000
eps = 0.1
NUM_TRIALS = 300
    
reward_average = np.zeros(RUN_LENGTH+1)
opt_action_average = np.zeros(RUN_LENGTH+1)

for j in range(NUM_TRIALS):
    reward_incremental= np.zeros(RUN_LENGTH+1)
    opt_action_incremental = np.zeros(RUN_LENGTH+1)

    mean = np.zeros(NUM_ARMS)
    var = np.ones(NUM_ARMS)
    qestimates = np.zeros(NUM_ARMS)
    qestimates_support = np.zeros(NUM_ARMS)
    for i in range(1,RUN_LENGTH+1):
        greedy = np.random.uniform(0,1) > eps
        arm = np.random.randint(0,NUM_ARMS)
        if greedy:
            arm = np.argmax(qestimates)
        r = np.random.normal(mean[arm],var[arm])
        cumulative_rew = reward_incremental[i-1]+(r-reward_incremental[i-1])/i

        reward_incremental[i] = cumulative_rew
        if np.argmax(mean) == arm:
            opt_action_incremental[i] = opt_action_incremental[i-1] + (1-opt_action_incremental[i-1])/i
        else:
            opt_action_incremental[i] = opt_action_incremental[i-1]*(i-1)/i
        qestimates_support[arm] +=1
        qestimates[arm] = qestimates[arm] + (r-qestimates[arm])/qestimates_support[arm]
        # Non-stationary 
        for i in range(NUM_ARMS):
            mean[i] += np.random.normal(0,0.01)
    reward_average = (reward_average*j + reward_incremental)/(j+1)
    opt_action_average = (opt_action_average*j + opt_action_incremental)/(j+1)


reward_alpha = np.zeros(RUN_LENGTH+1)
opt_action_alpha = np.zeros(RUN_LENGTH+1)

for j in range(NUM_TRIALS):
    reward_constant = np.zeros(RUN_LENGTH+1)
    opt_action_constant = np.zeros(RUN_LENGTH+1)

    mean = np.zeros(NUM_ARMS)
    var = np.ones(NUM_ARMS)*2
    qestimates = np.zeros(NUM_ARMS)
    alpha = 0.1

    for i in range(1,RUN_LENGTH+1):
        greedy = np.random.uniform(0,1) > eps
        arm = np.random.randint(0,NUM_ARMS)
        if greedy:
            arm = np.argmax(qestimates)
        r = np.random.normal(mean[arm],var[arm])
        cumulative_rew = reward_constant[i-1]+(r-reward_constant[i-1])/i
        reward_constant[i] = cumulative_rew
        if np.argmax(mean) == arm:
            opt_action_constant[i] = opt_action_constant[i-1] + (1-opt_action_constant[i-1])/i
        else:
            opt_action_constant[i] = opt_action_constant[i-1]*(i-1)/i
        qestimates[arm] = qestimates[arm] + (r-qestimates[arm])*alpha
        # Non-stationary 
        for i in range(NUM_ARMS):
            mean[i] += np.random.normal(0,0.01)
    reward_alpha = (reward_alpha*j + reward_constant)/(j+1)
    opt_action_alpha = (opt_action_alpha*j + opt_action_constant)/(j+1)


np.savetxt(sys.argv[1], (reward_average, opt_action_average, reward_alpha, opt_action_alpha))
