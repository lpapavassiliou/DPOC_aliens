import numpy as np
import scipy
from Constants import *
from ComputeTransitionProbabilities import probabilitySolver

def ComputeStageCosts(stateSpace, map_world, K):
    """
      Computes the stage costs
      for all states in the state space  for all control inputs.

    @type  stateSpace: (K x 4)-matrix
    @param stateSpace: Matrix where the i-th row represents the i-th
          element of the state space.

    @type  map_world: (M x N)-matrix
    @param map_world: A matrix describing the terrain.
              With values: FREE OBSTACLE PORTAL ALIEN MINE
              LAB BASE

    @type  K: integer
    @param K: Index representing the terminal state

    @return G:    A (K x L)-matrix containing the stage costs of all states in
                  the state space for all control inputs. The entry G(i, l)
                  represents the expected stage cost if we are in state i and
                  apply control input l.
    """
    G = np.ones((K, Constants.L))*np.inf

    solver = probabilitySolver(None, None, None, init=False)

    for stateIdx, state in enumerate(solver.stateSpace):
        if stateIdx == solver.labGemIdx:
            G[stateIdx][0] = 0.0
            G[stateIdx][1] = 0.0
            G[stateIdx][2] = 0.0
            G[stateIdx][3] = 0.0
            G[stateIdx][4] = 0.0
            continue

        for action in solver.getPossibleActions(state):
            nextState = solver.getNextState(state, action)
            nextStateIdx = solver.getStateIndex(nextState)

            if nextStateIdx in solver.portalsIdxes:
                nextState = solver.switchWorld(nextState)
                nextStateIdx = solver.getStateIndex(nextState)

            nAliens = 0
            if nextStateIdx in solver.aliensAreaIdxes:
                nAliens = solver.aliensAreaIdxes[nextStateIdx]

            G[stateIdx][action] = 1 + nAliens * Constants.N_a + solver.pDisturbed(nextStateIdx)
            sum_aliens = 0
            sum_out_or_obs = 0
            for disturbedAction in solver.getActions(nextState):
                disturbedNextState = solver.getNextState(nextState, disturbedAction)

                if solver.outOfBoundOrObstacle(disturbedNextState):
                    sum_out_or_obs += 1
                else:
                    disturbedNextStateIdx = solver.getStateIndex(disturbedNextState)
                    if disturbedNextStateIdx in solver.aliensAreaIdxes:
                        sum_aliens += solver.aliensAreaIdxes[disturbedNextStateIdx]
            G[stateIdx][action] += solver.pDisturbed(nextStateIdx)*(1/3)*(sum_aliens*Constants.N_a + sum_out_or_obs*Constants.N_b)

    return G