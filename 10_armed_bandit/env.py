# -*- coding: utf-8 -*-


'''
RL book 中总结的代码
env 类 环境
'''

import numpy as np
import matplotlib.pyplot as plt

class env_N_armed_test(object):
	def __init__(self, n_actions, base_value=0, noise=1):
		# action 的 q 值正态随机
		self.n_actions = n_actions
		self.actions = np.arange(n_actions)
		self.q_s = np.random.randn(self.n_actions) * 2 + base_value
		self.noise = noise
	
	def reset():
		self.q_s = np.random.randn(self.n_actions)
	
	
	def execute(self, action):
		reward = np.random.randn() * self.noise + self.q_s[action]
		return reward
		
	def plot_qs(self):
		markerline, stemlines, baseline = plt.stem(self.actions, self.q_s, '-.')
		plt.setp(baseline, color='b', linewidth=2)
		plt.xlabel('Action')
		plt.ylabel('Reward')
		plt.grid(True)
		plt.show()
		
		
class env_N_armed_test_nonst(env_N_armed_test):
	# nonstationary env where q_s varies during time
	# RL book chapter 2.5 exercise 2.5
	def execute(self, action):
		reward = np.random.randn() + self.q_s[action]
		self.q_s += np.random.randn(self.n_actions) * 0.01
		return reward