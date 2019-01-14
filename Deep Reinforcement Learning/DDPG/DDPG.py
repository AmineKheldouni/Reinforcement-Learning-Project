import tensorflow as tf
import numpy as np



class DDPG(object):
    def __init__(self, env, sess):
        self.sess = sess
        self.env = env.unwrapped
        self.s_dim = env.observation_space.shape[0]
        self.a_dim = env.action_space.shape[0]
        self.a_bound = env.action_space.high
        self.memory_capacity = 10000
        self.memory = np.zeros((self.memory_capacity, self.s_dim * 2 + self.a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()
        
        self.actor_lr = 0.001
        self.critic_lr = 0.002
        self.gamma = 0.9
        self.update_iter = 32
        self.max_episodes = 200
        self.max_steps = 200
        self.tau = 0.01
        self.moving_rewards = []

        
        self.S = tf.placeholder(tf.float32, [None, self.s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, self.s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        self.a = self._build_a(self.S,)
        q = self._build_c(self.S, self.a, )
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        ema = tf.train.ExponentialMovingAverage(decay=1 - self.tau)          # soft replacement

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]      # soft update operation
        a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)   # replaced target parameters
        q_ = self._build_c(self.S_, a_, reuse=True, custom_getter=ema_getter)

        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(self.actor_lr).minimize(a_loss, var_list=a_params)

        with tf.control_dependencies(target_update):    # soft replacement happened at here
            q_target = self.R + self.gamma * q_
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
            self.ctrain = tf.train.AdamOptimizer(self.critic_lr).minimize(td_error, var_list=c_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        indices = np.random.choice(self.memory_capacity, size=self.update_iter)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.memory_capacity  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)
    
    def work(self,render=True):
        var = 3  # control exploration. It will decrease along iterations.
        for i in range(self.max_episodes):
            s = self.env.reset()
            ep_reward = 0
            for j in range(self.max_steps):
                if render:
                    self.env.render()
        
                # Add exploration noise
                a = self.choose_action(s)
                a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration
                s_, r, done, info = self.env.step(a)
        
                self.store_transition(s, a, r / 10, s_)
        
                if self.pointer > self.memory_capacity:
                    var *= .9995    # decay the action randomness
                    self.learn()
        
                s = s_
                ep_reward += r
                if j == self.max_steps-1:
                    print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
                    if len(self.moving_rewards) == 0:  # record running episode reward
                        self.moving_rewards.append(ep_reward)
                    else:
                        self.moving_rewards.append(0.9 * self.moving_rewards[-1] + 0.1 * ep_reward)
                    break
