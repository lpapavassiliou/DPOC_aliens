import numpy as np
import scipy
from Constants import *

def ComputeTerminalStateIndex(stateSpace, map_world):
      """
      Computes the index of the terminal state in the stateSpace matrix

      @type  stateSpace: (K x 4)-matrix
      @param stateSpace: Matrix where the i-th row represents the i-th
            element of the state space.

      @type  map_world: (M x N)-matrix
      @param  map_world:      A matrix describing the terrain.
          With values: FREE OBSTACLE PORTAL ALIEN MINE
            LAB BASE

      @return stateIndex: An integer that is the index of the terminal state in the
              stateSpace matrix

      """
      stateCoord = np.where(map_world == Constants.LAB)
      endState = np.array([stateCoord[0][0], stateCoord[1][0], Constants.GEMS, Constants.UPPER])

      stateIndex = np.where((stateSpace == endState).all(axis=1))[0][0]

      return stateIndex
