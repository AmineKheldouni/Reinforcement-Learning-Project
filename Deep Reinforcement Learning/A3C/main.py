"""
Asynchronous Advantage Actor Critic (A3C) with continuous action space, Reinforcement Learning.

The Pendulum example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.8.0
gym 0.10.5
"""

import multiprocessing
import threading
import time

from A3C import *
from worker import Worker

nb_workers = multiprocessing.cpu_count()

env_name = 'Pendulum-v0'
env = gym.make(env_name)

session = tf.Session()
with tf.device("/cpu:0"):
    print('ok0')

    model = A3C(env)  # we only need its params
    workers = []
    print('ok')
    # Create worker
    for i in range(nb_workers):
        i_name = 'Worker_%i' % i   # worker name
        workers.append(Worker(i_name, model, env))

coordinator = tf.train.Coordinator()
session.run(tf.global_variables_initializer())

worker_threads = []
for worker in workers:
    job = lambda: worker.work(session, coordinator)
    t = threading.Thread(target=job)
    t.start()
    worker_threads.append(t)
coordinator.join(worker_threads)

avg_moving_reward = workers[-1].moving_rewards
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
    a = workers[-1].a3c_model.choose_action(s, session)
    s_next, r, done, info = env.step(a)

    cum_rewards += r
    s = s_next

session.close()
