# -*- coding: utf-8 -*-

'''
Monte Carlo control solving race track
'''

import numpy as np

class Track(object):
	def __init__(self, grid, start_line, finish_line):
		if not isinstance(grid, np.ndarray):
			grid = np.array(grid)
		if not isinstance(start_line, np.ndarray):
			start_line = np.array(start_line)
		if not isinstance(finish_line, np.ndarray):
			finish_line = np.array(finish_line)
		self.Track_Grid = grid
		self.Start_Line = start_line
		self.Finish_Line = finish_line
		self.grid_max = np.max(self.Track_Grid, axis=0)
		self.grid_min = np.min(self.Track_Grid, axis=0)
		self.X_Max = self.grid_max[0]
		self.X_Min = self.grid_min[0]
		self.Y_Max = self.grid_max[1]
		self.Y_Min = self.grid_min[1]
		
		# convert track grid(2d list or 2d np array) into a dict with keys as x and values as y
		self.Grid_X = set(p[0] for p in self.Track_Grid)
		self.Grid_Dict = {xs: set(p[1] for p in self.Track_Grid if p[0]==xs) for xs in self.Grid_X}
		
		# convert finish line(2d list or 2d np array) into a dict with keys as x and values as y
		self.Finish_X = set(p[0] for p in self.Finish_Line)
		self.Finish_Dict = {xs: set(p[1] for p in self.Finish_Line if p[0]==xs) for xs in self.Finish_X}
		
	def Within_Grid(self, p):
	#use dict may speed up the process
		if p[0] in self.Grid_Dict:
			return p[1] in self.Grid_Dict[p[0]]
		else: return False
		
	def Reach_Finish_Line(self, p):
	#use dict may speed up the process
		if p[0] in self.Finish_Dict:
			return p[1] in self.Finish_Dict[p[0]]
		else: return False
		