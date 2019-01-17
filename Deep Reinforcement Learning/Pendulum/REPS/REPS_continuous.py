# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 10:41:41 2019

@author: qduch
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import gym
import copy
import scipy
import scipy.optimize as opt


class Policy(object):
    def __init__(self,pi):
        n_states,n_actions = np.shape(pi)
        self.n_actions = n_actions
        self.n_states = n_states
        self.pi = pi
    def draw_action(self,state):
        u = np.random.rand()
        probas = np.cumsum(self.pi[state,:])
        a = 0
        while (a < self.n_actions-1 and (u > probas[a] or self.pi[state,a]==0)):
            a += 1
        return a
    

class RLModel:
    def __init__(self, env):
        self.env = env
        self.discrete_action = 20
        self.n_actions = 4*self.discrete_action+1
        self.discrete_state = 5
        self.vect_n_states = [0,(2*self.discrete_state+1),(2*self.discrete_state+1)*(16*self.discrete_state+1)]
        self.n_states = (2*self.discrete_state+1)*(2*self.discrete_state+1)*(16*self.discrete_state+1)
        self.ref = np.array([1,1,8])
        
        self.moving_rewards = []
        self.episode_count = 0
        
    def initialize_pi(self):
        pi = np.zeros((self.n_states,self.n_actions))
        for s in range(self.n_states):
            pi[s,:] = 1./self.n_actions * np.ones(self.n_actions)
        return pi

    def convert_state_to_int(self,notnormalize_next_state):
        next_state = 0
                
        for i in range(len(notnormalize_next_state)):
            next_state += self.vect_n_states[i]+ int( (notnormalize_next_state[i]+self.ref[i]) /self.discrete_state)
        return(next_state)
        
    def convert_int_to_action(self,action):
        return(-2+action/self.discrete_action)

    def collect_episodes(self, policy=None, horizon=None, n_episodes=1, render=False):
        paths = []

        for _ in range(n_episodes):
            observations = []
            actions = []
            rewards = []
            next_states = []
            ep_r = 0
            
            notnorm_state = self.env.reset()
            state = self.convert_state_to_int(notnorm_state)
            for t in range(horizon):
                action = policy.draw_action(state)
                notnorm_action = self.convert_int_to_action(action)
                notnormalize_next_state, reward, terminal, info = self.env.step(np.array([notnorm_action]))
                ep_r += reward
                next_state=self.convert_state_to_int(notnormalize_next_state)
                if render:
                    self.env.render()
                observations.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                notnorm_state = copy.copy(notnormalize_next_state)
                if terminal or t==horizon-1 :
                    # Finish rollout if terminal state reached
                    if len(self.moving_rewards) == 0:  # record running episode reward
                        self.moving_rewards.append(ep_r)
                    else:
                        self.moving_rewards.append(0.9 * self.moving_rewards[-1] + 0.1 * ep_r)
                    if self.episode_count%10==0:
                        print(
                            "| Episode:", self.episode_count,
                            "| Rewards: %i" % self.moving_rewards[-1],
                              )
                    self.episode_count += 1
                    break
                    # We need to compute the empirical return for each time step along the
                    # trajectory
            paths.append(dict(
                states=np.array(observations),
                actions=np.array(actions),
                rewards=np.array(rewards),
                next_states=np.array(next_states)
            ))
        return paths



class REPS(RLModel):
    def __init__(self, env):
        RLModel.__init__(self, env)
        self.p = 3
        self.N = 100
        self.eta = 0.1
        self.K = 50

    def compute_new_policy(self, eta, policy, phi, theta, samples):
        log_new_pi = np.zeros((policy.n_states,policy.n_actions))
        A = np.zeros((policy.n_states,policy.n_actions))
        counter = np.zeros((policy.n_states,policy.n_actions))
        nb_samples = 0
        for i in range(len(samples)):
            states = samples[i]['states']
            actions = samples[i]['actions']
            rewards = samples[i]['rewards']
            next_states = samples[i]['next_states']

            for j in range(len(states)):
                A[states[j],actions[j]] += rewards[j] + np.dot(phi[next_states[j],:],theta) - np.dot(phi[states[j],:],theta)
                counter[states[j],actions[j]] += 1
                nb_samples += 1
        for s in range(self.n_states):
            for a in range(self.n_actions):
                if counter[s,a]!=0:
                    A[s,a] /= counter[s,a]
        for s in range(policy.n_states):
            for a in range(policy.n_actions):
                argexpo = np.zeros(policy.n_actions)
                if policy.pi[s,a] == 0:
                    log_new_pi[s,a] = -float('inf')
                else:
                    for b in range(policy.n_actions):
                        argexpo[b] = np.log(policy.pi[s,b]+0.0001) + eta * A[s,b]
                    maxi = np.max(argexpo)
                    log_new_pi[s,a] = argexpo[a] - np.log(np.sum(np.exp(argexpo - maxi))) - maxi
        return(Policy(np.exp(log_new_pi)))


    def g(self, theta, eta, phi, samples):
        res = 0
        A = np.zeros((self.n_states,self.n_actions))
        counter = np.zeros((self.n_states,self.n_actions))
        nb_samples = 0
        for i in range(len(samples)):
            states = samples[i]['states']
            actions = samples[i]['actions']
            rewards = samples[i]['rewards']
            next_states = samples[i]['next_states']

            for j in range(len(states)):
                A[states[j],actions[j]] += rewards[j] + np.dot(phi[next_states[j],:],theta) - np.dot(phi[states[j],:],theta)
                counter[states[j],actions[j]] += 1
                nb_samples += 1
        for s in range(self.n_states):
            for a in range(self.n_actions):
                if counter[s,a]!=0:
                    A[s,a] /= counter[s,a]
        for i in range(len(samples)):
            states = samples[i]['states']
            actions = samples[i]['actions']
            for j in range(len(states)):
                res += np.exp(eta*A[states[j],actions[j]])
        res /= nb_samples
        return (np.log(res)/eta)

    def Dg(self, theta, eta, phi, samples):
        n_states,p = np.shape(phi)
        numerator = 0
        denominator = 0
        A = np.zeros((self.n_states,self.n_actions))
        D = np.zeros((self.n_states,self.n_actions,p))
        counter = np.zeros((self.n_states,self.n_actions))
        for i in range(len(samples)):
            states = samples[i]['states']
            actions = samples[i]['actions']
            rewards = samples[i]['rewards']
            next_states = samples[i]['next_states']

            for j in range(len(states)):
                A[states[j],actions[j]] += rewards[j] + np.dot(phi[next_states[j],:],theta) - np.dot(phi[states[j],:],theta)
                D[states[j],actions[j],:] += phi[next_states[j],:] - phi[states[j],:]
                counter[states[j],actions[j]] += 1
        for s in range(self.n_states):
            for a in range(self.n_actions):
                if counter[s,a]!=0:
                    A[s,a] /= counter[s,a]
        for s in range(self.n_states):
            for a in range(self.n_actions):
                if counter[s,a]!=0:
                    D[s,a,:] /= counter[s,a]
        for i in range(len(samples)):
            states = samples[i]['states']
            actions = samples[i]['actions']
            for j in range(len(states)):
                numerator += np.exp(eta*A[states[j],actions[j]]) * D[states[j],actions[j]]
                denominator += np.exp(eta*A[states[j],actions[j]])
        return ((1/eta) * numerator / denominator)

    def compute_phi(self,p):
        phi = np.zeros((self.n_states,p))
        for k in range(self.n_states):
            phi[k,:] = [k,k**2,np.log(k+1)]
        return phi

    def update(self):
        """Relative Entropy Policy Search using Mirror Descent"""
        p = 3    
        # initialization of the distribution
        pi = self.initialize_pi()
        policy = Policy(pi)
        #Tmax =  -100*np.log(10e-6)/(1-env.gamma)
        T = 100
        theta = [0 for i in range(p)]
        phi = self.compute_phi(p)
        
        for k in tqdm(range(self.K), desc="Iterating REPS algorithm..."):
            ##### SAMPLING
            samples = self.collect_episodes(policy=policy,horizon=T,n_episodes=self.N)

            #### OPTIMIZE
            theta = opt.fmin_bfgs(self.g,x0=theta,fprime=self.Dg,args=(self.eta,phi,samples), disp=0)

            #### COMPUTE THE NEW POLICY
            policy = self.compute_new_policy(self.eta,policy,phi,theta,samples) 
            
        self.policy = policy
        self.theta = theta
        self.phi = phi
        
env_name = 'Pendulum-v0'
env = gym.make(env_name)
REPS_model = REPS(env)
REPS_model.update()

avg_moving_reward = REPS_model.moving_rewards
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
    a = REPS_model.policy.draw_action(REPS_model.convert_state_to_int(s))
    s_nex, r, done, info = env.step(REPS_model.convert_int_to_action(a))
    s_next = REPS_model.convert_state_to_int(s_nex)

    cum_rewards += r
    s = s_next