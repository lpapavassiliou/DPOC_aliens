import numpy as np
import scipy
from Constants import *
from typing import List
from ComputeTerminalStateIndex import *
from MakePlots import *
from collections import namedtuple
"""
  ^
  |       
  |y
  |        x
(0,0) ---------->

SOUTH = 0
NORTH = 1
WEST = 2
EAST = 3
STAY = 4
"""

solver = None
class probabilitySolver:
      actionDictionary = {0: np.array([0, -1, 0, 0]), 1: np.array([0, 1, 0, 0]), 2: np.array([-1, 0, 0, 0]), 3: np.array([1, 0, 0, 0]), 4: np.array([0, 0, 0, 0])}
      StateKey = namedtuple('StateKey', ['x', 'y', 'gems', 'world'])
      stateActionKey = namedtuple('stateActionKey', ['state', 'action'])
      possibleActionDict = {}

      map_world = None
      stateSpace = None
      K = None
      stateDict = None
      stateActionDict = None
      P = None
      baseNoGemIdx = None
      labGemIdx = None
      mineNoGemIdx = None
      mineGemIdx = None
      aliensIdxes = None
      portalsIdxes = None
      aliensAreaIdxes = None

      UPPER_TRIANGLE = True
      LOWER_TRIANGLE = False

      def __init__(self, map_world, stateSpace, K, init=True) -> None:
            if init:
                  probabilitySolver.map_world = map_world
                  probabilitySolver.stateSpace = stateSpace
                  probabilitySolver.K = K
                  probabilitySolver.stateDict = {}
                  probabilitySolver.stateActionDict = {}
                  probabilitySolver.P = np.zeros((self.K, self.K, Constants.L))

                  probabilitySolver.baseNoGemIdx = None
                  probabilitySolver.labGemIdx = None
                  probabilitySolver.mineNoGemIdx = None
                  probabilitySolver.mineGemIdx = None
                  probabilitySolver.aliensIdxes = []
                  probabilitySolver.portalsIdxes = []
                  probabilitySolver.aliensAreaIdxes = {}

                  self.createStateDict()
                  probabilitySolver.aliensAreaIdxes = self.getAliensAreaIdx()

      
      def calculateProbability(self):
            [self.probabilityTask(item) for item in np.arange(0, len(self.stateSpace))]
            return self.P
      
      def createStateDict(self):
            for idx, state in enumerate(self.stateSpace):
                  key = self.StateKey(state[0], state[1], state[2], state[3])
                  self.stateDict[key] = idx

                  stateType = self.map_world[state[0]][state[1]]
                  gems = state[2]
                  world = state[3]

                  if stateType != Constants.FREE:
                        if stateType == Constants.BASE and gems == Constants.EMPTY and world == Constants.UPPER:
                              probabilitySolver.baseNoGemIdx = idx
                        elif stateType == Constants.LAB and gems == Constants.GEMS and world == Constants.UPPER:
                              probabilitySolver.labGemIdx = idx
                        elif stateType == Constants.MINE and gems == Constants.EMPTY and world == Constants.LOWER:
                              probabilitySolver.mineNoGemIdx = idx
                        elif stateType == Constants.MINE and gems == Constants.GEMS and world == Constants.LOWER:
                              probabilitySolver.mineGemIdx = idx
                        elif stateType == Constants.ALIEN and world == Constants.LOWER:
                              probabilitySolver.aliensIdxes.append(idx)
                        elif stateType == Constants.PORTAL:
                              probabilitySolver.portalsIdxes.append(idx)

      @staticmethod
      def getTriangleDirection(state):
          return ((state[0] + state[1] + state[3]) % 2) == 0

      def getNextState(self, state: np.ndarray, action: int) -> np.ndarray:
            #TODO: transform into dictionary
            return state + self.actionDictionary[action]

      def getStateIndex(self,state):
            return self.stateDict[self.StateKey(state[0], state[1], state[2], state[3])]

      def getPossibleActions(self, state: np.ndarray) -> List[int]:
            initialPossibleActions = [Constants.SOUTH, Constants.NORTH, Constants.WEST, Constants.EAST, Constants.STAY]

            if state == probabilitySolver.stateSpace[probabilitySolver.labGemIdx]:
                  return initialPossibleActions
                  
            triangleDirection = probabilitySolver.getTriangleDirection(state)
            # Δ or y = N-1
            if triangleDirection == probabilitySolver.UPPER_TRIANGLE or state[1] == Constants.N-1:
                  initialPossibleActions.remove(Constants.NORTH)
            # ∇ or y = 0
            if triangleDirection == probabilitySolver.LOWER_TRIANGLE or state[1] == 0:
                  initialPossibleActions.remove(Constants.SOUTH)
            # x = 0
            if state[0] == 0:
                  initialPossibleActions.remove(Constants.WEST)
            # x = M-1
            elif state[0] == Constants.M-1:
                  initialPossibleActions.remove(Constants.EAST)

            # check if hits obstacle
            for action in initialPossibleActions.copy():
                  nextState = self.getNextState(state, action)
                  if self.map_world[nextState[0]][nextState[1]] == Constants.OBSTACLE:
                        initialPossibleActions.remove(action)

            return initialPossibleActions

      def getActions(self, state: np.ndarray) -> List[int]:
            initialPossibleActions = [Constants.SOUTH, Constants.NORTH, Constants.WEST, Constants.EAST]
            triangleDirection = probabilitySolver.getTriangleDirection(state)
            # Δ 
            if triangleDirection == probabilitySolver.UPPER_TRIANGLE:
                  initialPossibleActions.remove(Constants.NORTH)
            # ∇ 
            if triangleDirection == probabilitySolver.LOWER_TRIANGLE:
                  initialPossibleActions.remove(Constants.SOUTH)

            return initialPossibleActions

      def getAliensAreaIdx(self):
            res = {}
            for alienIdx in self.aliensIdxes:
                  actions = self.getPossibleActions(self.stateSpace[alienIdx])
                  for action in actions:
                        state = self.getNextState(self.stateSpace[alienIdx], action)
                        stateIdx = self.getStateIndex(state)

                        if stateIdx not in res:
                              res[stateIdx] = 1
                        else:
                              res[stateIdx] += 1
            return res

      def outOfBoundOrObstacle(self, state):
            return state[0] < 0 or state[1] < 0 or state[0] > Constants.M-1 or state[1] > Constants.N-1 or self.map_world[state[0]][state[1]] == Constants.OBSTACLE

      def looseGems(self, state):
            return [state[0], state[1], Constants.EMPTY, state[3]]

      def switchWorld(self, state):
            return [state[0], state[1], state[2], 1-state[3]]

      def pDisturbed(self, stateIdx):
            return Constants.P_DISTURBED * ( 1 + (Constants.S-1.0)*self.stateSpace[stateIdx][3])

      def calculateProbNoGems(self, stateIdx):
            probDict = {}

            probDict[stateIdx] = 1.0 - self.pDisturbed(stateIdx)

            state = self.stateSpace[stateIdx]

            for disturbedAction in self.getActions(state):
                  disturbedNextState = self.getNextState(state, disturbedAction)
                  disturbedNextStateIdx = None

                  # if going into an obstacle or out of bound
                  if self.outOfBoundOrObstacle(disturbedNextState):
                        disturbedNextStateIdx = self.baseNoGemIdx
                        if disturbedNextStateIdx not in probDict:
                              probDict[disturbedNextStateIdx] = self.pDisturbed(stateIdx) / 3.0
                        else:
                              probDict[disturbedNextStateIdx] += self.pDisturbed(stateIdx) / 3.0
                  else:
                        disturbedNextStateIdx = self.getStateIndex(disturbedNextState)

                        # 1a) If in a portal, switch world
                        if disturbedNextStateIdx in self.portalsIdxes:
                              disturbedNextState = self.switchWorld(disturbedNextState)
                              disturbedNextStateIdx = self.getStateIndex(disturbedNextState)

                        if disturbedNextStateIdx == self.mineNoGemIdx:
                              disturbedNextStateIdx = self.mineGemIdx

                        if disturbedNextStateIdx not in probDict:
                              probDict[disturbedNextStateIdx] = self.pDisturbed(stateIdx) / 3.0
                        else:
                              probDict[disturbedNextStateIdx] += self.pDisturbed(stateIdx) / 3.0


            return probDict

      def calculateProbWithGems(self, stateIdx):
            probDict = {}

            probDict[stateIdx] = 1.0 - self.pDisturbed(stateIdx)

            state = self.stateSpace[stateIdx]

            for disturbedAction in self.getActions(state):
                  disturbedNextState = self.getNextState(state, disturbedAction)
                  disturbedNextStateIdx = None

                  # if going into an obstacle or out of bound
                  if self.outOfBoundOrObstacle(disturbedNextState):
                        disturbedNextStateIdx = self.baseNoGemIdx
                        if disturbedNextStateIdx not in probDict:
                              probDict[disturbedNextStateIdx] = self.pDisturbed(stateIdx) / 3.0
                        else:
                              probDict[disturbedNextStateIdx] += self.pDisturbed(stateIdx) / 3.0
                  else:
                        disturbedNextStateIdx = self.getStateIndex(disturbedNextState)

                        # 1a) If in a portal, switch world
                        if disturbedNextStateIdx in self.portalsIdxes:
                              disturbedNextState = self.switchWorld(disturbedNextState)
                              disturbedNextStateIdx = self.getStateIndex(disturbedNextState)

                        if disturbedNextStateIdx not in self.aliensAreaIdxes or disturbedNextStateIdx == self.mineNoGemIdx:
                              # I certainly retain the gems
                              probDict[disturbedNextStateIdx] = self.pDisturbed(stateIdx) / 3.0
                        else:
                              # first case: I retain the gems
                              probDict[disturbedNextStateIdx] = (self.pDisturbed(stateIdx) / 3.0)*(Constants.P_PROTECTED**self.aliensAreaIdxes[disturbedNextStateIdx])
                              # second case: I loose them
                              noGemDisturbedNextStateIdx = self.getStateIndex(self.looseGems(disturbedNextState))
                              probDict[noGemDisturbedNextStateIdx] = (self.pDisturbed(stateIdx) / 3.0)*(1.0-(Constants.P_PROTECTED**self.aliensAreaIdxes[disturbedNextStateIdx]))


            return probDict

      def insertStateProb(self, stateIdx, action, nextStateIdx, prob):
            key = probabilitySolver.stateActionKey(stateIdx, action)
            
            if key not in probabilitySolver.stateActionDict:
                  probabilitySolver.stateActionDict[key] = {}
                  probabilitySolver.stateActionDict[key][nextStateIdx] = prob
            else:
                  if nextStateIdx not in probabilitySolver.stateActionDict[key]:
                        probabilitySolver.stateActionDict[key][nextStateIdx] = prob
                  else:
                        probabilitySolver.stateActionDict[key][nextStateIdx] += prob

      def probabilityTask(self, stateIdx):
            state = self.stateSpace[stateIdx]
            # When in the lab, stay in the lab
            if stateIdx == self.labGemIdx:
                  self.P[stateIdx][stateIdx][0] = 1.0
                  self.P[stateIdx][stateIdx][1] = 1.0
                  self.P[stateIdx][stateIdx][2] = 1.0
                  self.P[stateIdx][stateIdx][3] = 1.0
                  self.P[stateIdx][stateIdx][4] = 1.0

                  self.insertStateProb(stateIdx, 0, stateIdx, 1.0)
                  self.insertStateProb(stateIdx, 1, stateIdx, 1.0)
                  self.insertStateProb(stateIdx, 2, stateIdx, 1.0)
                  self.insertStateProb(stateIdx, 3, stateIdx, 1.0)
                  self.insertStateProb(stateIdx, 4, stateIdx, 1.0)

                  probabilitySolver.possibleActionDict[stateIdx] = [0,1,2,3,4]
                  return
            
            for action in self.getPossibleActions(state):
                  # Append to possibleActionDict
                  if stateIdx not in probabilitySolver.possibleActionDict:
                        probabilitySolver.possibleActionDict[stateIdx] = [action]
                  else:
                        probabilitySolver.possibleActionDict[stateIdx].append(action)

                  # Go to the desired place with the desired action
                  nextState = self.getNextState(state, action)
                  nextStateIdx = self.getStateIndex(nextState)

                  # 1a) If in a portal, switch world
                  if nextStateIdx in self.portalsIdxes:
                        nextState = self.switchWorld(nextState)
                        nextStateIdx = self.getStateIndex(nextState)

                  # 1c) If in the mine with no gems, add gems
                  if nextStateIdx ==self. mineNoGemIdx:
                        nextStateIdx = self.mineGemIdx
                        nextState[2] = Constants.GEMS

                        probDict = self.calculateProbWithGems(nextStateIdx)

                        for keyState in probDict:
                              self.P[stateIdx][keyState][action] += probDict[keyState]
                              self.insertStateProb(stateIdx, action, keyState, probDict[keyState])

                  else:
                        if state[2] != Constants.GEMS:
                              probDict = self.calculateProbNoGems(nextStateIdx)

                              for keyState in probDict:
                                    self.P[stateIdx][keyState][action] += probDict[keyState]
                                    self.insertStateProb(stateIdx, action, keyState, probDict[keyState])
                        else:
                              if nextStateIdx not in self.aliensAreaIdxes:
                                    probDict = self.calculateProbWithGems(nextStateIdx)
                                    for keyState in probDict:
                                          self.P[stateIdx][keyState][action] += probDict[keyState]
                                          self.insertStateProb(stateIdx, action, keyState, probDict[keyState])
                              else:
                                    probDictProtected = self.calculateProbWithGems(nextStateIdx)
                                    noGemNextStateIdx = self.getStateIndex(self.looseGems(nextState))
                                    probDictLost = self.calculateProbNoGems(noGemNextStateIdx)

                                    for keyState in probDictProtected:
                                          self.P[stateIdx][keyState][action] += (Constants.P_PROTECTED**self.aliensAreaIdxes[nextStateIdx]) * probDictProtected[keyState]
                                          self.insertStateProb(stateIdx, action, keyState, (Constants.P_PROTECTED**self.aliensAreaIdxes[nextStateIdx]) * probDictProtected[keyState])
                                    for keyState in probDictLost:
                                          self.P[stateIdx][keyState][action] += (1.0 - (Constants.P_PROTECTED**self.aliensAreaIdxes[nextStateIdx])) * probDictLost[keyState]
                                          self.insertStateProb(stateIdx, action, keyState, (1.0 - (Constants.P_PROTECTED**self.aliensAreaIdxes[nextStateIdx])) * probDictLost[keyState])

def  ComputeTransitionProbabilities(stateSpace, map_world, K):
      """
      Computes the transition probabilities between all states in the state space for
      all control inputs.

      @type  stateSpace: (K x 4)-matrix
      @param stateSpace: Matrix where the i-th row represents the i-th
            element of the state space.

      @type  map_world: (M x N)-matrix
      @param  map_world:      A matrix describing the terrain.
            With values: FREE OBSTACLE PORTAL ALIEN MINE
            LAB BASE

      @type  K: integer
      @param K: An integer representing the total number of states in the state space

      @return P:
                A (K x K x L)-matrix containing the transition probabilities
                between all states in the state space for all control inputs.
                The entry P(i, j, l) represents the transition probability
                from state i to state j if control input l is applied.

      """
      solver = probabilitySolver(map_world, stateSpace, K)
      return solver.calculateProbability()
