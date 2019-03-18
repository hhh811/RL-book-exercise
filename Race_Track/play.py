# -*- coding: utf-8 -*-

'''
Monte Carlo control solving race track
'''

import numpy as np
from env import *
from agent import *


def Main():
    # initialize env and agent
    # grid = [
    #			[0, 0],
    #			[1, 0],
    #			[2, 0],
    #			[0, 1],
    #			[1, 1],
    #			[2, 1],
    #			[0, 2],
    #			[1, 2],
    #			[2, 2],
    #			[0, 3],
    #			[1, 3],
    #			[2, 3],
    #			[0, 4],
    #			[1, 4],
    #			[2, 4],
    #			[3, 4],
    #			[4, 4],
    #			[5, 4],
    #			[6, 4],
    #			[7, 4],
    #			[0, 5],
    #			[1, 5],
    #			[2, 5],
    #			[3, 5],
    #			[4, 5],
    #			[5, 5],
    #			[6, 5],
    #			[7, 5],
    #			[0, 6],
    #			[1, 6],
    #			[2, 6],
    #			[3, 6],
    #			[4, 6],
    #			[5, 6],
    #			[6, 6],
    #			[7, 6],
    #			]
    #	start_line = [
    #			[0, 0],
    #			[1, 0],
    #			[2, 0]
    #			]
    #	finish_line = [
    #			[7, 4],
    #			[7, 5],
    #			[7, 6]
    #			]
    grid = [
        [0, 0],
        [1, 0],
        [2, 0]
    ]
    start_line = [
        [0, 0]
    ]
    finish_line = [
        [2, 0]
    ]
    track = Track(grid, start_line, finish_line)
    agent = Agent(track)
    gamma = 0.9

    # initialize global variables
    Record = []  # list of agent record of states, actions and rewards_record
    Summary = {}  # dict of episode: steps taken if successful or 'Failed' if failed to reach finish line

    for i in range(10):
        # Generate an episode according to b policy
        # return a list of (Si, Ai, Ri+1)
        # print('episode %s start' % (i+1))
        result = agent.run()
        if result:
            Record.append((agent.states_record, agent.actions_record, agent.rewards_record))
            agent.update_values(gamma)
        else:
            Record.append('Failed')
        Summary[i + 1] = result
        if i % 50 == 0:
            print('episode %s done ' % (i + 1))
            print('------ \n')

    # print steps taken of successful episode, should be descending if the algorithm is right
    print('Summary: steps taken')
    for v in Summary.values():
        if v: print(v)
    print(agent.Q_s_a.values())


if __name__ == '__main__':
    Main()
