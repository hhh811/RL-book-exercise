# -*- coding: utf-8 -*-

'''
Dynamic Programming solving Jack's car Rental problem
'''

import numpy as np


class Car_Rental():
	def __init__(self, discount_rate):
		#self.s = np.asarray((20, 20))    # cars at 1st location
		
		'''
		policy is a map from (s1, s2) to -5~5
		use a 2d matrix to define Policy
		'''
		self.P = np.zeros((21, 21), dtype=np.int32)
		
		# value function is similar
		self.V = np.zeros((21, 21))
		
		# lambda value
		self.discount_rate = discount_rate
		
	def step(self, s, action):
		'''
		calculate the distribution of next_s, r over s and action
		calculate the update of state values
		'''
		v_update = np.zeros_like(self.V)
		#cars rental requests and returned
		for rr1 in range(10):     # cars request in 1st location
			p_rr1 = 3 ** rr1 * np.exp(-3) / np.math.factorial(rr1)
			for rr2 in range(10):     # cars request in 2nd location
				p_rr2 = 4 ** rr2 * np.exp(-4) / np.math.factorial(rr2)
				for cr1 in range(10):     # cars returned in 1st location
					p_cr1 = 3 ** cr1 * np.exp(-3) / np.math.factorial(cr1)
					for cr2 in range(10):     # cars returned in 2nd location
						p_cr2 = 2 ** cr2 * np.exp(-2) / np.math.factorial(cr2)
						# accumulate possibilities of next_s
						rent = np.minimum(s, np.array((rr1, rr2)))   # cars actually rent
						next_s = np.clip(s - rent, 0, 20)    #rent out, cars over 20 will be removed
						next_s = np.clip(s + np.array((cr1, cr2)), 0, 20)    #return back, cars over 20 will be removed
						posb = p_rr1 * p_rr2 * p_cr1 * p_cr2
						
						cars_moved = np.maximum(np.minimum(action, s[0]), -s[1])    # cars moved cannot be more than cars at original location
						next_s += np.asarray((-cars_moved, cars_moved), dtype=np.int32)
						next_s = np.clip(next_s, 0, 20)
						
						#revenue every day
						r = 10 * np.sum(rent) - 2 * np.abs(cars_moved)
						
						#calculate the update of state values
						v_update[tuple(next_s)] += posb * (r + self.discount_rate * self.V[tuple(next_s)])				
		return v_update

	def Policy_Evaluation(self, theta=1):
		iter = 0
		while True:
			delta = 0
			iter += 1
			# iterate over state
			for i in range(21):
				for j in range(21):
					s = np.array((i, j))
					v = self.V[tuple(s)]
					# iterate over actions (initially no cars are moved)	
					action = self.P[tuple(s)]
					v_update = self.step(s, action)
					self.V += v_update
					delta = max(delta, abs(v - self.V[tuple(s)]))
					print(i, j)
			print('Iter %d done!' % iter)
			if delta < theta or iter > 10: return
			
			
	def Policy_Improvement(self):
		policy_stable = True
		# iterate over state
		for i in range(21):
			for j in range(21):
				s = np.array((i, j))
				old_action = self.P[tuple(s)]
				best_action = -5
				best_value = 0
				#get the action improved
				for k in range(11):
					action = k - 5    # -5~5
					v_update = self.step(s, action)
					if v_update > best_value: best_action, best_value = action, v_update
					self.self.P[tuple(s)] = best_action
				if old_action != self.self.P[tuple(s)]: policy_stable = False
		if policy_stable: return
		else: self.Policy_Evaluation()

		
def play():
	car_rental = Car_Rental(discount_rate=0.9)
	car_rental.Policy_Evaluation()
	car_rental.Policy_Improvement()
	print(car_rental.V)

	
if __name__ == '__main__':
	play()