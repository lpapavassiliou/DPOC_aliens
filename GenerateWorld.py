# uncompyle6 version 3.8.0
# Python bytecode 3.8.0 (3413)
# Decompiled from: Python 3.8.13 (default, Mar 28 2022, 11:38:47) 
# [GCC 7.5.0]
# Embedded file name: C:\Users\utente\Documents\ETH\Autumn_semester_2022-2023\DPOC\DPOC_ProgEx22\DPOC_PE_2022_Python\GenerateWorld.py
# Compiled at: 2022-11-06 14:42:34
# Size of source mod 2**32: 7935 bytes
from random import randrange
import numpy as np
from Constants import *

def GenerateWorld(width, height):
    """
        Input arguments:

               width:
                   Integer describing the width of the map, M.

               height:
                   Integer describing the length of the map, N.

        Output arguments:

                map:
               A (M x N) matrix describing the terrain of the map. map(m,n)
               represents the cell at indices (m,n) according to the axes
               specified in the PDF.
   """
    if isinstance(width, int):
        isinstance(height, int) or print('Warning: Width or height not an integer!')
        width = int(width)
        height = int(height)
        print(f"New width: {width}")
        print(f"New height: {height}")
    else:
        if width < 8 or height < 8:
            print('Error: Minimum width and height is 8!')
            print('Exiting function')
            return None
        obstacleDensity = 0.06
        obstacleScalingWidth = 1.0
        obstacleScalingHeight = 0.4
        portalDensity = 0.01
        alienDensity = 0.03
        feasible = False
        while not feasible:
            map = np.zeros((width, height))
            for k in range(round(obstacleDensity * width * height)):
                obstacleCenter = np.array([round(np.random.uniform() * (width - 1)),
                 round(np.random.uniform() * (height - 1))])
                obstacleSize = np.array([round(abs(np.random.normal()) * obstacleScalingWidth + 1),
                 round(abs(np.random.normal()) * obstacleScalingHeight + 1)])
                mLow = max(0, round(obstacleCenter[0] - obstacleSize[0] / 2))
                mHigh = min(width - 1, round(obstacleCenter[0] + obstacleSize[0] / 2))
                nLow = max(0, round(obstacleCenter[1] - obstacleSize[1] / 2))
                nHigh = min(height - 1, round(obstacleCenter[1] + obstacleSize[1] / 2))
                map[mLow:mHigh, nLow:nHigh] = Constants.OBSTACLE
            else:
                for k in range(round(portalDensity * width * height)):
                    while True:
                        portal = np.array([round(np.random.uniform() * (width - 1)),
                         round(np.random.uniform() * (height - 1))])
                        if map[(portal[0], portal[1])] == Constants.FREE:
                            break

                    map[(portal[0], portal[1])] = Constants.PORTAL
                else:
                    for k in range(round(alienDensity * width * height)):
                        while True:
                            alien = np.array([round(np.random.uniform() * (width - 1)),
                             round(np.random.uniform() * (height - 1))])
                            if map[(alien[0], alien[1])] == Constants.FREE:
                                break

                        map[(alien[0], alien[1])] = Constants.ALIEN
                    else:
                        while True:
                            mine = np.array([round(np.random.uniform() * (width - 1)),
                             round(np.random.uniform() * (height - 1))])
                            if map[(mine[0], mine[1])] == Constants.FREE:
                                break

                        map[(mine[0], mine[1])] = Constants.MINE
                        while True:
                            lab = np.array([round(np.random.uniform() * (width - 1)),
                             round(np.random.uniform() * (height - 1))])
                            if map[(lab[0], lab[1])] == Constants.FREE:
                                break

                        map[(lab[0], lab[1])] = Constants.LAB
                        while True:
                            base = np.array([round(np.random.uniform() * (width - 1)),
                             round(np.random.uniform() * (height - 1))])
                            r = randrange(4)
                            if r == 0:
                                base[0] = 0
                            else:
                                if r == 1:
                                    base[1] = 0
                                else:
                                    if r == 2:
                                        base[0] = width - 1
                                    else:
                                        if r == 3:
                                            base[1] = height - 1
                            if map[(base[0], base[1])] == Constants.FREE:
                                break

                        map[(base[0], base[1])] = Constants.BASE
                        feasible = check_map(map, lab)

    return map


def check_map(map, start):
    feasible_normal = False
    feasible_inverted = False
    stack = [
     start]
    visited_cells = np.zeros_like(map)
    visited_cells[(start[0], start[1])] = 1
    if stack:
        current_cell = stack.pop()
        unvisited_neighbors = find_unvisited_neighbors_normal(current_cell, visited_cells)
        for i in range(len(unvisited_neighbors)):
            cell = unvisited_neighbors[i]
            if map[(cell[0], cell[1])] == Constants.OBSTACLE:
                pass
            else:
                visited_cells[(cell[0], cell[1])] = 1
                stack.append(cell)

    else:
        if visited_cells.sum() == (map != Constants.OBSTACLE).sum():
            feasible_normal = True
        stack = [
         start]
        visited_cells = np.zeros_like(map)
        visited_cells[(start[0], start[1])] = 1
        while stack:
            current_cell = stack.pop()
            unvisited_neighbors = find_unvisited_neighbors_inverted(current_cell, visited_cells)
            for i in range(len(unvisited_neighbors)):
                cell = unvisited_neighbors[i]
                if map[(cell[0], cell[1])] == Constants.OBSTACLE:
                    pass
                else:
                    visited_cells[(cell[0], cell[1])] = 1
                    stack.append(cell)

    if visited_cells.sum() == (map != Constants.OBSTACLE).sum():
        feasible_inverted = True
    return feasible_normal and feasible_inverted


def find_unvisited_neighbors_normal(current_cell, visited_cells):
    unvisited_neighbors = []
    m = current_cell[0]
    n = current_cell[1]
    if m % 2 == n % 2:
        if n - 1 >= 0:
            if visited_cells[(m, n - 1)] == 0:
                unvisited_neighbors.append([m, n - 1])
    if m % 2 != n % 2:
        if n + 1 < visited_cells.shape[1]:
            if visited_cells[(m, n + 1)] == 0:
                unvisited_neighbors.append([m, n + 1])
    if m - 1 >= 0:
        if visited_cells[(m - 1, n)] == 0:
            unvisited_neighbors.append([m - 1, n])
    if m + 1 < visited_cells.shape[0]:
        if visited_cells[(m + 1, n)] == 0:
            unvisited_neighbors.append([m + 1, n])
    return unvisited_neighbors


def find_unvisited_neighbors_inverted(current_cell, visited_cells):
    unvisited_neighbors = []
    m = current_cell[0]
    n = current_cell[1]
    if m % 2 != n % 2:
        if n - 1 >= 0:
            if visited_cells[(m, n - 1)] == 0:
                unvisited_neighbors.append([m, n - 1])
    if m % 2 == n % 2:
        if n + 1 < visited_cells.shape[1]:
            if visited_cells[(m, n + 1)] == 0:
                unvisited_neighbors.append([m, n + 1])
    if m - 1 >= 0:
        if visited_cells[(m - 1, n)] == 0:
            unvisited_neighbors.append([m - 1, n])
    if m + 1 < visited_cells.shape[0]:
        if visited_cells[(m + 1, n)] == 0:
            unvisited_neighbors.append([m + 1, n])
    return unvisited_neighbors
# okay decompiling GenerateWorld.pyc
