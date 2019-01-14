import time
import gym
import matplotlib.pyplot as plt
from DDPG import *

env_name = 'Pendulum-v0'
env = gym.make(env_name)
tf.reset_default_graph() 
session = tf.Session()
model = DDPG(env, session)
model.work()

avg_moving_reward = model.moving_rewards
plt.plot(np.arange(len(avg_moving_reward)), avg_moving_reward)
plt.xlabel('Step')
plt.ylabel('Total moving reward')
plt.show()


s = env.reset()
cum_rewards = 0
done = False
while not done:
    env.render()
    time.sleep(0.1)
    a = model.choose_action(s, session)
    s_next, r, done, info = env.step(a)

    cum_rewards += r
    s = s_next

session.close()