#export PYTHONPATH=$PYTHONPATH:~/cog_ml_tasks/:~/cog_tasks_rl_agents/

import gym
import sys
import torch
import random

sys.path.insert(1, 'dmn_pytorch')
sys.path.insert(1, 'gym-babi/gym_babi/envs')

from babi_env import BabiEnv
from DRQN.DRQN_agent import Agent_DRQN
from common.utils import train, test, save_train_res, train_results_plots, load_train_res

N_tr = 10000
N_tst = 5000

TASK_ID = 1
MAX_MEM_SIZE = 400 #300
LR = 1e-3
EPSILON = 0.999
GAMMA = 0.9
PARAMS = {
    "lstm_hidden_size": 50,
    "n_lstm_layers": 2, #1
    "linear_hidden_size": 50,
    "n_linear_layers": 1
}

env = BabiEnv(TASK_ID)
env_test = BabiEnv(TASK_ID, mode='test')

agent = Agent_DRQN(len(env.state_space), len(env.action_space), MAX_MEM_SIZE, LR, \
                    EPSILON, GAMMA, PARAMS)
                    
res_tr = train(env, agent, N_tr, seed=123, print_progress=False, render=False)
res_te = test(env_test, agent, N_tst, seed=123, print_progress=False)

save_train_res('./results/{0}_drqn_tr2'.format(TASK_ID), res_tr)
save_train_res('./results/{0}_drqn_te2'.format(TASK_ID), res_te)

te1, te2, te3 = load_train_res('./results/{0}_drqn_te2.npy'.format(TASK_ID))
res_tr = load_train_res('./results/{0}_drqn_tr2.npy'.format(TASK_ID))

train_results_plots(dir='./plots/', figname='{0}_tr'.format(TASK_ID), names=['DRQN_tr'], \
                    numbers=[res_tr])

print('Plots saved for task', TASK_ID)