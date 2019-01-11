# -*- coding: utf-8 -*-

'''
RL book 中总结的代码
执行程序
'''

import numpy as np
import matplotlib.pyplot as plt

from agents import *
from env import *


def  play():
	# hyper parameters
	N_steps = 5000
	# first several steps are randomly chosen to accumulate memory
	N_rand_steps = 0
	# epsilon
	epsilon_list = [0.1]

	# param for tuning
	#param = agent
	
	# initialize env and agent
	N_actions = 10
	env = env_N_armed_test(N_actions, base_value=4.0, noise=0.0)
	actions = env.actions
	#agent = Agent(actions, epsilon=0)
	agent_list = [Agent_GBA(actions, with_baseline=False), Agent_GBA(actions, with_baseline=False), 
					Agent_GBA(actions, with_baseline=False), Agent_GBA(actions, alpha = 0.4, with_baseline=False)]
	
	# accumulate data for plotting results
	Total_reward = 0
	Average_reward = []
	N_Optimal_action = 0   # number of correctly choose optimal action
	Optimal_action = []    # list of percentage of correctly choose optimal action
	Results_summary = {agent: {'Average_reward': None, 'Optimal_action': None} for agent in agent_list}
	
	print(env.q_s)
	
	#for epsilon in epsilon_list:
	for agent in agent_list:
		#agent.reset(epsilon = epsilon)
		next_action = np.random.choice(actions)  # used when N_rand_steps is set 0
		for i in range(N_steps):
			if i < N_rand_steps:
				action = np.random.choice(actions)
				reward, next_action = agent.step(action, env)
			else:
				reward, next_action = agent.step(next_action, env)
			Total_reward += reward
			Average_reward.append(Total_reward/(i + 1))
			optimal_action = np.argmax(env.q_s)
			N_Optimal_action += int(next_action == optimal_action)
			Optimal_action.append(N_Optimal_action/(i + 1))
		Results_summary[agent]['Average_reward'] = Average_reward
		Results_summary[agent]['Optimal_action'] = Optimal_action
		print('Total Reward: %4d' % Total_reward)
		Total_reward = 0
		N_Optimal_action = 0
		Average_reward = []
		Optimal_action = [] 
		
	
	env.plot_qs()
	
	plt.figure()
	x = np.arange(N_steps) + 1
	plt.subplot(2, 1, 1)
	plt.xlabel('Steps')
	plt.ylabel('Average_reward')
	plt.grid(True)
	#for epsilon in epsilon_list:
	for agent in agent_list:
		#plt.plot(x, Results_summary[epsilon]['Average_reward'], label='epsilon = %s' % epsilon)
		plt.plot(x, Results_summary[agent]['Average_reward'], label='agent = %s' % agent.__class__)
	plt.legend()
	
	plt.subplot(2, 1, 2)
	plt.xlabel('Steps')
	plt.ylabel('Optimal_action')
	plt.grid(True)
	#for epsilon in epsilon_list:
	for agent in agent_list:
		#plt.plot(x, Results_summary[epsilon]['Optimal_action'], label='epsilon = %s' % epsilon)
		plt.plot(x, Results_summary[agent]['Optimal_action'], label='agent = %s' % agent.__class__)
	plt.legend()

	plt.show()
		
		
if __name__ == '__main__':
	play()