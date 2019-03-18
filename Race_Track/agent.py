# -*- coding: utf-8 -*-

'''
Monte Carlo control solving race track
'''

import numpy as np
from env import *


class Agent(object):
	def __init__(self, track):
		self.V = np.array([0, 0])
		self.Track = track
		self.P = track.Start_Line[np.random.choice(len(track.Start_Line))]
		self.Actions = np.array([
						[-1, -1],
						[-1, 0],
						[-1, 0],
						[0, -1],
						[0, 0],
						[0, 1],
						[1, -1],
						[1, 0],
						[1, 1]
						])
		self.states_record = []
		self.actions_record = []
		self.rewards_record = []
		
		self.Q_s_a = {}
		self.C_s_a = {}
		
	def step(self, action):
		finished = False
		next_reward = -1
		self.states_record.append(self.P)   # record states before action taken
		self.V = self.V + action
		#print('\n last position:%s, v:%s, action:%s' % (self.P, self.V, action))
		self.P = self.P + self.V
		#print('after action %s' % self.P)
		if not self.Track.Within_Grid(self.P):
			self.P = self.Track.Start_Line[np.random.choice(len(self.Track.Start_Line))]
			#print('hit! replace to %s' % self.P)
			self.V = np.array([0, 0])
		if self.Track.Reach_Finish_Line(self.P):    #注意这里numpy数组的比较 不能简单用 in
			next_reward = 1
			finished = True
			#print('!!! Finished !!!')
			#print('finished position: %s' % self.P)
			#print('finish line: %s' %self.Track.Finish_Line)
		
		self.rewards_record.append(next_reward)
		return finished
		
	def reset(self):
		self.V = np.array([0, 0])
		self.P = self.Track.Start_Line[np.random.choice(len(self.Track.Start_Line))]
		#print('initial position %s' % self.P)
		self.states_record = []
		self.actions_record = []
		self.rewards_record = []
	
	def run(self):
		#print('start line %s' % self.Track.Start_Line)
		self.reset()
		result = False
		#while True:
		for i in range(999):
			qs_temp = self.get_qs_temp(self.P)
			action = greedy_policy(qs_temp, self.Actions)
			self.actions_record.append(action)
			finished = self.step(action)
			if finished: 
				print('!!! Finished !!!  %s steps taken' % (i+1))
				result = i+1
				break
		#print('Too many steps, didn`t reach')
		return result
			
	def update_values(self, gamma):
		G = 0
		W = 1
		for i in range(len(self.states_record)):
			step = len(self.states_record) - 1 - i
			G = gamma * G + self.rewards_record[step]
			s = self.states_record[step]
			a = self.actions_record[step]
			s_a = (tuple(s), tuple(a))
			self.C_s_a[s_a] = self.C_s_a.get(s_a, 0) + W
			self.Q_s_a[s_a] = self.Q_s_a.get(s_a, 0) + W / (self.C_s_a.get(s_a, 0) + 0.01) * (G - self.Q_s_a.get(s_a, 0))
			qs_temp = self.get_qs_temp(s)
			pi_st = greedy_policy(qs_temp, self.Actions, epsilon=0, TIES_BROKEN='CONSISTENT')
			if any(pi_st != a): break
			b_at_st = greedy_policy(qs_temp, self.Actions, action_taken=a, TIES_BROKEN='CONSISTENT')
			W = W / (b_at_st + 0.0001)
			
	def get_qs_temp(self, s):
		qs_temp = []
		for i in range(len(self.Actions)):
			a = self.Actions[i]
			s_a = (tuple(s), tuple(a))
			q_s_a = self.Q_s_a.get(s_a, 0)
			qs_temp.append(q_s_a)
		return qs_temp
		


def greedy_policy(qs, actions, action_taken=None, epsilon=0.1, TIES_BROKEN='RANDOM'):
	# choose action according to greedy policy
	# q is a list/array of q_values of 9 actions
	# if action_taken is not None, calculate and return the conditional possibility b(At|St)
	if not isinstance(qs, np.ndarray): qs = np.array(qs)
	p = np.random.rand()
	if p < epsilon:
		next_action = actions[np.random.choice(len(actions))]
	else:
		# 多个最大值， 按一定规则选
		if TIES_BROKEN == 'CONSISTENT':
			next_action = actions[np.argmax(qs)]
		# 多个最大值， 随机选
		elif TIES_BROKEN == 'RANDOM':
			next_action = actions[np.random.choice(np.where(qs == np.amax(qs))[0])]
	
	if action_taken is not None:
		p_at_st = 1 - epsilon / len(actions)
		if any(next_action != action_taken): p_at_st = epsilon / len(actions)
		return p_at_st
	else:
		return next_action