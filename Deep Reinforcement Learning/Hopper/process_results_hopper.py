import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

window = int(sys.argv[1])

##############################################################
################## LIFE  #####################################
##############################################################


results_a3c = pd.read_csv('./Mujoco_A3C_hopper/trace.csv',
                      sep=',',
                      header=0)
results_ppo = pd.read_csv('./Mujoco_ClippedPPO_hopper/trace.csv',
                      sep=',',
                      header=0)
results_ddpg = pd.read_csv('./Mujoco_DDPG_hopper/trace.csv',
                      sep=',',
                      header=0)

variable = "Episode Length"
time_col = "Episode #"

results_a3c[[variable]] = results_a3c[[variable]].rolling(window=window).mean()
results_ppo[[variable]] = results_ppo[[variable]].rolling(window=window).mean()
results_ddpg[[variable]] = results_ddpg[[variable]].rolling(window=window).mean()

ax0 = results_a3c.plot(x=time_col, y=variable, color='red')
results_ppo.plot(x=time_col, y=variable, ax=ax0, color='blue')
results_ddpg.plot(x=time_col, y=variable, ax=ax0, color='green')
ax0.legend(["A3C", "Clipped PPO", "DDPG"])
plt.title('Life through episodes (moving average over ' + str(window) + ' episodes)')
plt.savefig('agent_life_hopper.png')
plt.show()
plt.clf()
##############################################################
################## TRAINING REWARD #########################
##############################################################

results_a3c = pd.read_csv('./Mujoco_A3C_hopper/trace.csv',
                      sep=',',
                      header=0)
results_ppo = pd.read_csv('./Mujoco_ClippedPPO_hopper/trace.csv',
                      sep=',',
                      header=0)
results_ddpg = pd.read_csv('./Mujoco_DDPG_hopper/trace.csv',
                      sep=',',
                      header=0)

variable = "Training Reward"

results_a3c[[variable]] = results_a3c[[variable]].rolling(window=window).mean()
results_ppo[[variable]] = results_ppo[[variable]].rolling(window=window).mean()
results_ddpg[[variable]] = results_ddpg[[variable]].rolling(window=window).mean()

ax0 = results_a3c.plot(x=time_col, y=variable, color='red')
results_ppo.plot(x=time_col, y=variable, ax=ax0, color='blue')
results_ddpg.plot(x=time_col, y=variable, ax=ax0, color='green')
ax0.legend(["A3C", "Clipped PPO", "DDPG"])
plt.title('Life through episodes (moving average over ' + str(window) + ' episodes)')
plt.savefig('training_reward_hopper.png')
plt.show()
plt.clf()
