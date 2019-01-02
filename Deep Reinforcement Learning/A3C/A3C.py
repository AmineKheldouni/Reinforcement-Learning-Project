
import tensorflow as tf
import numpy as np
import gym
import os
import matplotlib.pyplot as plt

class A3C(object):
    def __init__(self, env, scope='GlobalNet', a3c_model=None):
        self.state_shape = env.observation_space.shape[0]
        self.action_shape = env.action_space.shape[0]
        self.action_bounds = [env.action_space.low, env.action_space.high]

        self.actor_lr = 0.0001    # learning rate for actor
        self.critic_lr = 0.001    # learning rate for critic
        self.beta = 0.01

        if scope == 'GlobalNet':   # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, self.state_shape], 'S')
                self.a_params, self.c_params = self._build_net(scope)[-2:]
        else:   # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, self.state_shape], 'S')
                self.a_his = tf.placeholder(tf.float32, [None, self.action_shape], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                mu, sigma, self.v, self.a_params, self.c_params = self._build_net(scope)

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

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, a3c_model.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, a3c_model.c_params)]
                with tf.name_scope('push'):
                    actor_optimizer = tf.train.RMSPropOptimizer(self.actor_lr, name='RMSPropA')
                    critic_optimizer = tf.train.RMSPropOptimizer(self.critic_lr, name='RMSPropC')
                    self.update_a_op = actor_optimizer.apply_gradients(zip(self.a_grads, a3c_model.a_params))
                    self.update_c_op = critic_optimizer.apply_gradients(zip(self.c_grads, a3c_model.c_params))

    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
            mu = tf.layers.dense(l_a, self.action_shape, tf.nn.tanh, kernel_initializer=w_init, name='mu')
            sigma = tf.layers.dense(l_a, self.action_shape, tf.nn.softplus, kernel_initializer=w_init, name='sigma')
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return mu, sigma, v, a_params, c_params

    def update_global(self, feed_dict, session):  # run by a local
        session.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self, session):  # run by a local
        session.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s, session):  # run by a local
        s = s[np.newaxis, :]
        return session.run(self.A, {self.s: s})
