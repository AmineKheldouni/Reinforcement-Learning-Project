import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

window = int(sys.argv[1])
env_name = str(sys.argv[2])
algorithm_name = str(sys.argv[3])

##############################################################
################## LIFE  #####################################
##############################################################


# results = pd.read_csv('./' + env_name + '/' + algorithm_name + '_results/trace.csv',
#                       sep=',',
#                       header=0)
#
# variable = "Episode Length"
# time_col = "Episode #"
#
# results = results[[time_col,variable]].dropna()
# results[[variable]] = results[[variable]].rolling(window=window).mean()
#
# results.plot(x=time_col, y=variable, color='blue')
# plt.title('Life through episodes (moving average over ' + str(window) + ' episodes)')
# plt.savefig('agent_life_'+env_name+'.png')
# plt.show()
# plt.clf()
##############################################################
################## TRAINING REWARD #########################
##############################################################

results = pd.read_csv('./' + env_name + '/' + algorithm_name + '_results/trace.csv',
                      sep=',',
                      header=0)
time_col = "Episode #"
variable = "Training Reward"

results = results[[time_col,variable]].dropna()
results[[variable]] = results[[variable]].rolling(window=window).mean()

results.plot(x=time_col, y=variable, color='green')
plt.title('Training Reward through episodes')
plt.savefig('./' + env_name + '/training_reward_' +env_name + '_' + algorithm_name + '.png')
plt.show()
plt.clf()
