# -*- coding: utf-8 -*-


'''
RL book 中总结的代码
agent 类
'''

import numpy as np

class Agent(object):
	def __init__(self, actions, epsilon = 0.1):
		self.actions = actions
		self.N_action_taken = np.zeros(len(self.actions))   # Total steps taken of each action in an epoch
		self.action_values = np.zeros(len(self.actions))
		# greedy factor
		self.epsilon = epsilon
	
	
	def reset(self, epsilon = 0.1):
		#self.RT_list = []
		#self.AT_list = []
		self.N_action_taken = np.zeros(len(self.actions))
		self.action_values = np.zeros(len(self.actions))
		self.epsilon = epsilon
	
	
	def step(self, action, env):
		# agent take an action and receives reward from env
		reward = env.execute(action)
		
		# update action values
		self.N_action_taken[action] += 1
		self.update_action_value(action, reward)
		
		# make decision about next action
		next_action = self.choose_action()
		return reward, next_action
		
	def update_action_value(self, action, reward):
		# using average sample mean
		# RL book chapter 2.4 equation 2.3
		self.action_values[action] += (reward - self.action_values[action]) / self.N_action_taken[action]
		
	def choose_action(self):
		next_action = greedy_policy(self.actions, self.action_values, self.epsilon)
		return next_action


class Agent_alpha(Agent):
	# rewrite rule for update action values
	# RL book chapter 2.5
	def __init__(self, actions, alpha = 0.1):
		super().__init__(actions)
		self.alpha = alpha
	
	def update_action_value(self, action, reward):
		self.action_values[action] += self.alpha * (reward - self.action_values[action])

		
class Agent_UCB(Agent):
	# rewrite rule for choose next_action
	# RL book chapter 2.6
	def __init__(self, actions, alpha=0.1, c=2):
		super().__init__(actions)
		self.c = c
		r.avg = 0
		
	def choose_action(self):
		next_action = UCB_policy(self.actions, self.action_values, self.N_action_taken, self.c)
		return next_action
		
		
class Agent_GBA(Agent):
	# gradient bandit algorithms
	# RL book chapter 2.8
	def __init__(self, actions, alpha=0.1, with_baseline=True):
		super().__init__(actions)
		self.alpha = alpha
		# 2 more variables H(preference) and pi(policy)
		self.H = np.zeros(len(self.actions))
		self.update_pi()
		self.N_steps = 0
		self.ave_reward = 0
		self.with_baseline = with_baseline
		
	def step(self, action, env):
		# agent take an action and receives reward from env
		reward = env.execute(action)
		self.N_steps += 1
		if self.with_baseline: self.ave_reward += (reward - self.ave_reward) / self.N_steps
		self.update_H(action, reward)
		self.update_pi()
		
		# make decision about next action
		next_action = self.choose_action()
		#print('step: %s' % self.N_steps)
		#print('action: %s reward: %s ave_reward: %s\nH: %s\npi: %s' % (action, reward, self.ave_reward, self.H, self.pi))
		#print('=' * 20 + '\n')
		return reward, next_action
	
	def choose_action(self):
		p = np.random.rand()
		if p < self.epsilon:
			next_action = np.random.choice(self.actions)
		else:
			next_action = np.random.choice(self.actions, p=self.pi)
		
		return next_action
	
	def update_H(self, action, reward):
		self.H = self.H + self.alpha * (reward - self.ave_reward) * ((self.actions == action) - self.pi)
			
	def update_pi(self):
		self.pi = np.exp(self.H) / np.sum(np.exp(self.H))
		
		
def greedy_policy(actions, action_values, epsilon):
	# choose action according to greedy policy
	p = np.random.rand()
	if p < epsilon:
		next_action = np.random.choice(actions)
	elif np.sum(action_values == np.amax(action_values)) > 1:
		# 如果有多个最大值， 从中随机选
		next_action = np.random.choice(np.where(action_values == np.amax(action_values))[0])
	else:
		next_action = np.argmax(action_values)
	
	return next_action


def UCB_policy(actions, action_values, N_action_taken, c):
	t = np.sum(N_action_taken)
	assert t > 0
	next_action = np.argmax(action_values + c * np.sqrt(np.log(t) / (N_action_taken + 0.1)))
	return next_action

