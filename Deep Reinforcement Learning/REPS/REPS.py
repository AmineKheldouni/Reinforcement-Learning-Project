import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
import scipy
from scipy import optimize as opt

class REPS(object):
    def __init__(self, env, sess):
        self.env = env
        self.dim_phi = 9

        self.state_shape = env.observation_space.shape[0]
        self.action_shape = env.action_space.shape[0]
        self.action_bounds = [env.action_space.low, env.action_space.high]

        self.moving_rewards = []

        self.episode_count = 0
        self.max_episodes = 2000
        self.max_steps = 500
        self.update_iter = 32 #batch size
        self.gamma = 0.9

        self.lr = 0.0001    # learning rate
        self.ksi = 0.001

        self.s = tf.placeholder(tf.float32, [None, self.state_shape], 'S')

        self.logprob = []

        # critic
        self.r = tf.placeholder(tf.float32, [None, 1], 'rewards')

        # actor
        pi, pi_params = self._build_net('pi', trainable=True)
        self.sample_op = tf.squeeze(pi.sample(1), axis=0)  # operation of choosing action

        oldpi, oldpi_params = self._build_net('oldpi', trainable=False)
        self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.a = tf.placeholder(tf.float32, [None, self.action_shape], 'action')

        self.phis = tf.placeholder(tf.float32, [None, self.dim_phi], 'phis')
        self.aloss = tf.reduce_mean( tf.exp(tf.tensordot(self.phis,self.theta,0) / self.ksi),1)

        self.atrain_op = tf.train.AdamOptimizer(self.lr).minimize(self.aloss)
        sess.run(tf.global_variables_initializer())




        self.s = tf.placeholder(tf.float32, [None, self.state_shape], 'S')
        self.a_his = tf.placeholder(tf.float32, [None, self.action_shape], 'A')
        self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

        mu, sigma, self.v, self.params = self._build_net()

        td = tf.subtract(self.v_target, self.v, name='TD_error')
        with tf.name_scope('c_loss'):
            self.c_loss = tf.reduce_mean(tf.square(td))

        with tf.name_scope('wrap_a_out'):
            mu, sigma = mu * self.action_bounds[1], sigma + 1e-4

        normal_dist = tf.distributions.Normal(mu, sigma)

        with tf.name_scope('a_loss'):
            log_prob = normal_dist.log_prob(self.a_his)
            exp_v = log_prob * tf.stop_gradient(td)
            entropy = normal_dist.entropy()  # encourage exploration
            self.exp_v = self.beta * entropy + exp_v
            self.a_loss = tf.reduce_mean(-self.exp_v)

        with tf.name_scope('choose_a'):  # use local params to choose action
            self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=[0, 1]), self.action_bounds[0], self.action_bounds[1])
        with tf.name_scope('local_grad'):
            self.a_grads = tf.gradients(self.a_loss, self.a_params)
            self.c_grads = tf.gradients(self.c_loss, self.c_params)

        with tf.name_scope('pull'):
            self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.params, sess.params)]
        with tf.name_scope('push'):
            critic_optimizer = tf.train.RMSPropOptimizer(self.lr, name='RMSPropC')
            self.update_c_op = critic_optimizer.apply_gradients(zip(self.c_grads, sess.c_params))

    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .1)
        l_c = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
        v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return v, params

    def update_global(self, feed_dict, session):  # run by a local
        session.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self, session):  # run by a local
        session.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s, session):  # run by a local
        s = s[np.newaxis, :]
        return session.run(self.A, {self.s: s})














    def update(self, feed_dict, sess):
        if self.episode_count < self.max_episodes:
            sess.run(self.update_oldpi_op)     # copy pi to old pi
            s, a, r, phis = feed_dict['states'], feed_dict['actions'], feed_dict['rewards'], feed_dict['phis']

            self.optimize(s,a,r)

            sess.run(self.atrain_op, {self.s: s, self.a: a, self.phis: phis})

            sess.run([self.update_oldpi_op])


    def phi(self,s):
        res = []
        for i in range(self.state_shape):
            res.append(s[i])
            res.append(np.cos(s[i]))
            res.append(np.sin(s[i]))
        return np.array(res)


    # value. Initialize dual function g(\theta, v). \eta > 0
    # First eval delta_v
    def optimize(self,s,a,r):
        def f_dual(x):
            ksi = x[0]
            theta = x[1:]
            buffer_bellman_error_target = []
            vnew = np.dot(self.phi(s[0]),theta)
            for i in range(len(r)-1):
                vold = vnew
                vnew = np.dot(self.phi(s[i+1]),theta)
                bell_err = r[i] + self.gamma * vnew - vold
                buffer_bellman_error_target.append(bell_err)
            res = ksi * np.log( np.mean(np.exp(buffer_bellman_error_target / ksi) ) )
            return (res)

        def f_dual_grad(x):
            ksi = x[0]
            theta = x[1:]
            buffer_bellman_error_target = []
            buffer_bellman_phi = []

            vnew = np.dot(self.phi(s[0,:]),theta)
            for i in range(len(r)-1):
                vold = vnew
                vnew = np.dot(self.phi(s[i+1,:]),theta)
                bell_err = r[i] + self.gamma * vnew - vold
                buffer_bellman_error_target.append(bell_err)

                bell_phi = self.gamma * self.phi(s[i+1,:]) - self.phi(s[i,:])
                buffer_bellman_phi.append(bell_phi)
            dtheta = np.zeros(self.dim_phi)
            buffer_bellman_error_target = np.array(buffer_bellman_error_target)
            buffer_bellman_phi = np.array(buffer_bellman_phi)
            for i in range(self.dim_phi):
                dtheta += np.mean(buffer_bellman_phi[:,i] *  np.exp(buffer_bellman_error_target / ksi))
            dtheta = ksi * dtheta
            dtheta /= np.mean(np.exp(buffer_bellman_error_target / ksi))

            dksi = np.log( np.mean(np.exp(buffer_bellman_error_target / ksi) ) )
            dksi -= (1/ksi**2) * np.mean(buffer_bellman_error_target * np.exp(buffer_bellman_error_target / ksi) ) / np.mean( np.exp(buffer_bellman_error_target / ksi) )

            return np.hstack([dksi, dtheta])


        # Initial BFGS parameter values.
        x0 = np.hstack([self.ksi, self.theta])

        # Set parameter boundaries: \ksi>0, theta unrestricted.
        bounds = [(-np.inf, np.inf) for _ in x0]
        bounds[0] = (0., np.inf)

        # Optimize through BFGS
        ksi_before = x0[0]
        dual_before = f_dual(x0)
        opti_res = opt.minimize(f_dual, x0, method='L-BFGS-B', jac=f_dual_grad,
            bounds=bounds)
        params_ast = opti_res.x
        dual_after = f_dual(params_ast)

        # Optimal values have been obtained
        self.ksi = params_ast[0]
        self.theta = params_ast[1:]
        print(ksi_before,self.ksi)
        print('theta',self.theta)


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
                    buffer_phis = []
                    for i in range(len(buffer_s)):
                        buffer_phis.append(self.phi(buffer_s[i]))

                    buffer_s, buffer_a, buffer_phis = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_phis)
                    feed_dict = {
                        'states': buffer_s,
                        'actions': buffer_a,
                        'rewards': buffer_r,
                        'phis': buffer_phis,
                    }
                    self.update(feed_dict, session)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.env.render()
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
