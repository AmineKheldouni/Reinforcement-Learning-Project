import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym

class PPOKL(object):
    def __init__(self, env, sess):
        self.env = env
        self.state_shape = env.observation_space.shape[0]
        self.action_shape = env.action_space.shape[0]
        self.action_bounds = [env.action_space.low, env.action_space.high]

        self.moving_rewards = []

        self.episode_count = 0
        self.max_episodes = 2500
        self.max_steps = 500
        self.update_iter = 32 #batch size
        self.gamma = 0.9

        self.actor_lr = 0.0001    # learning rate for actor
        self.critic_lr = 0.001    # learning rate for critic

        self.kl_target = 0.003
        self.beta = 1
        self.beta_max = 20
        self.beta_min = 1/20

        self.s = tf.placeholder(tf.float32, [None, self.state_shape], 'S')

        # critic
        l_c = tf.layers.dense(self.s, 100, tf.nn.relu)
        self.v = tf.layers.dense(l_c, 1)
        self.v_target = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.advantage = self.v_target - self.v
        self.closs = tf.reduce_mean(tf.square(self.advantage))
        self.ctrain_op = tf.train.AdamOptimizer(self.critic_lr).minimize(self.closs)

        # actor
        pi, pi_params = self._build_net('pi', trainable=True)
        oldpi, oldpi_params = self._build_net('oldpi', trainable=False)
        self.sample_op = tf.squeeze(pi.sample(1), axis=0)  # operation of choosing action
        self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.a = tf.placeholder(tf.float32, [None, self.action_shape], 'action')
        self.adv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
        ratio = pi.prob(self.a) / (oldpi.prob(self.a) + 1e-5)

        self.kl_divergence = tf.reduce_mean(tf.contrib.distributions.kl_divergence(pi, oldpi))


        self.aloss = -tf.reduce_mean(ratio * self.adv)
        self.aloss += self.beta * self.kl_divergence

        self.atrain_op = tf.train.AdamOptimizer(self.actor_lr).minimize(self.aloss)
        sess.run(tf.global_variables_initializer())

    def update(self, feed_dict, sess):
        if self.episode_count < self.max_episodes:
            sess.run(self.update_oldpi_op)     # copy pi to old pi
            s, a, r = feed_dict['states'], feed_dict['actions'], feed_dict['v_target']
            adv = sess.run(self.advantage, {self.s: s, self.v_target: r})
            # update actor and critic in a update loop
            sess.run(self.atrain_op, {self.s: s, self.a: a, self.adv: adv})
            sess.run(self.ctrain_op, {self.s: s, self.v_target: r})
            kl_divergence = sess.run(self.kl_divergence, {self.s: s, self.v_target: r})
            if kl_divergence < self.kl_target / 1.5:
                self.beta /= 2
            elif kl_divergence > self.kl_target * 1.5:
                self.beta *= 2
            self.beta = np.clip(self.beta, self.beta_min, self.beta_max)
            sess.run([self.update_oldpi_op])


    def _build_net(self, name, trainable):
        with tf.variable_scope(name):
            l_a = tf.layers.dense(self.s, 200, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.layers.dense(l_a, self.action_shape, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l_a, self.action_shape, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s, sess):
        s = s[np.newaxis, :]
        a = sess.run(self.sample_op, {self.s: s})[0]
        return np.clip(a, -2, 2)

    def get_v(self, s, sess):
        s = s[np.newaxis, :]
        return sess.run(self.v, {self.s: s})[0, 0]

    def work(self, session):
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while self.episode_count < self.max_episodes:
            s = self.env.reset()
            ep_r = 0
            for ep_t in range(self.max_steps):
                a = self.choose_action(s, session)
                s_next, r, done, info = self.env.step(a)
                done = True if ep_t == self.max_steps - 1 else False

                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append((r + 8) / 8)                    # normalize reward, find to be useful
                if total_step % self.update_iter == 0 or done:   # update global and assign to local net
                    if done:
                        v_s_ = 0   # terminal
                    else:
                        v_s_ = self.get_v(s_next, session)
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + self.gamma * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        'states': buffer_s,
                        'actions': buffer_a,
                        'v_target': buffer_v_target,
                    }
                    self.update(feed_dict, session)
                    buffer_s, buffer_a, buffer_r = [], [], []
                s = s_next
                total_step += 1
                if done:
                    if len(self.moving_rewards) == 0:  # record running episode reward
                        self.moving_rewards.append(ep_r)
                    else:
                        self.moving_rewards.append(0.9 * self.moving_rewards[-1] + 0.1 * ep_r)
                    if self.episode_count%10==0:
                        print(
                            "Worker: ", 'Worker'
                            "| Episode:", self.episode_count,
                            "| Rewards: %i" % self.moving_rewards[-1],
                              )
                    self.episode_count += 1
                    break
