import time

from PPOKL import *

t0 = time.time()
env_name = 'Pendulum-v0'
env = gym.make(env_name)
tf.reset_default_graph() 
session = tf.Session()
model = PPOKL(env, session)  # we only need its params
model.work(session)

avg_moving_reward = model.moving_rewards
plt.plot(np.arange(len(avg_moving_reward)), avg_moving_reward)
plt.xlabel('Step')
plt.ylabel('Total moving reward')
plt.savefig('PPO_KL_moving_rwd')
plt.show()


print("Computational time: ",time.time()-t0)


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