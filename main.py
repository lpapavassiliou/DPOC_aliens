import scipy.io
import numpy as np
import matplotlib.pyplot as plt

from ComputeStageCosts import *
from ComputeTerminalStateIndex import *
from ComputeTransitionProbabilities import *
from GenerateWorld import *
from MakePlots import *

from Constants import *
import SolutionPolicyIteration
import SolutionValueIteration
import SolutionHybridIteration
import SolutionMDPToolBox
import SolutionLP

import time


#1: Value iteration
#2: MDP Toolbox
#3: ValueIteration+PolicyIteration


solution = 3

if __name__ == "__main__":
    """
    Set to true to generate a random map of size mapSize, else set to false
    to load the pre-existing example map
    """
    generateRandomWorld = False

    """
    Generate map
    map(m,n) represents the cell type at indices (m,n) according to the axes
    specified in the PDF.
    """

    print('Generate map')
    if generateRandomWorld:
        map_world = GenerateWorld(Constants.M, Constants.N)
    else:
        # We can load a pre-generated map_world
        data = scipy.io.loadmat('samples/exampleWorld_1.mat')
        P_given = scipy.io.loadmat('samples/exampleP_1.mat')
        G_given = scipy.io.loadmat('samples/exampleG_1.mat')
        sol_P = P_given["P"]
        sol_G = G_given["G"]
        map_world = data["map"]
        print(map_world.shape)

    """
     Generate state space
     Generate a (K x 4)-matrix 'stateSpace', where each accessible cell is
     represented by 4 rows.
    """

    print('Generate state space')
    stateSpace = []
    for m in range(0, len(map_world)):
        for n in range(0, len(map_world[0])):
            if map_world[m][n] != Constants.OBSTACLE:
                stateSpace.extend([[m, n, Constants.EMPTY, Constants.UPPER],
                                   [m, n, Constants.GEMS, Constants.UPPER],
                                   [m, n, Constants.EMPTY, Constants.LOWER],
                                   [m, n, Constants.GEMS, Constants.LOWER]])

    # State space size
    K = len(stateSpace)

    # Set the following to True as you progress with the files
    terminalStateIndexImplemented = True
    transitionProbabilitiesImplemented = True
    stageCostsImplemented = True
    SolutionImplemented = True

    # Compute the terminal state index
    if terminalStateIndexImplemented:

        # Done
        TERMINAL_STATE_INDEX = ComputeTerminalStateIndex(stateSpace, map_world)
    else:
        TERMINAL_STATE_INDEX = None

    # Compute transition probabilities

    now = time.time()
    if transitionProbabilitiesImplemented:
        print('Compute transition probabilities')

        """
            Compute the transition probabilities between all states in the
            state space for all control inputs.
            The transition probability matrix has the dimension (K x K x L), i.e.
            the entry P(i, j, l) represents the transition probability from state i
            to state j if control input l is applied.
        """

        # TODO: Question b)
        P = ComputeTransitionProbabilities(stateSpace, map_world, K)
    else:
        P = np.zeros((K, K, Constants.L))
    print("It took: ", round(time.time()-now, 4), " secs")
    print("Correct solution" if np.array_equal(sol_P,P) else "Wrong solution")

    # Compute stage costs
    now = time.time()
    if stageCostsImplemented:
        print("Compute stage costs")

        """
            Compute the stage costs for all states in the state space for all
            control inputs.
            The stage cost matrix has the dimension (K x L), i.e. the entry G(i, l)
            represents the cost if we are in state i and apply control input l.
        """

        # TODO: Question c)
        G = ComputeStageCosts(stateSpace, map_world, K)
    else:
        G = np.ones((K, Constants.L))*np.inf
        
    print("It took: ", round(time.time()-now, 4), " secs")
    print("Correct solution" if np.allclose(sol_G,G) else "Wrong solution")

    # Solve the stochastic shortest path problem
    now = time.time()
    if SolutionImplemented:
        print('Solve stochastic shortest path problem')

        # TODO: Question d)
        if solution == 0:
            print('--- Pseudo Policy Iteration selected ---')
            [J_opt, u_opt_ind] = SolutionPolicyIteration.Solution(P, G, K, TERMINAL_STATE_INDEX)
        elif solution == 1:
            print('--- Value Iteration selected ---')
            [J_opt, u_opt_ind] = SolutionValueIteration.Solution(P, G, K, TERMINAL_STATE_INDEX)
        elif solution == 2:
            print('--- MDPToolBox selected ---')
            [J_opt, u_opt_ind] = SolutionMDPToolBox.Solution(P, G, K, TERMINAL_STATE_INDEX)
        elif solution == 3:
            print('--- HybridIteration selected ---')
            [J_opt, u_opt_ind] = SolutionHybridIteration.Solution(P, G, K, TERMINAL_STATE_INDEX)
        elif solution == 4:
            print('--- LP selected ---')
            [J_opt, u_opt_ind] = SolutionLP.Solution(P, G, K, TERMINAL_STATE_INDEX)

        if len(J_opt) != K or len(u_opt_ind) != K:
            print('[ERROR] the size of J and u must be K')
        else:
            print("It took: ", round(time.time()-now, 4), " secs")
            print('Evaluating your policy...')

            # SUBMITTED POLICY EVALUATION
            if True:
                ps = probabilitySolver(None,None,None, init=False)
                V = np.zeros(K)
                epsilon = 0
                looping = True
                while looping:
                    max_deltaV = 0
                    for state_idx in range(K):
                        if state_idx != ps.labGemIdx:
                            action = u_opt_ind[state_idx]
                            curr_exp_value = 0.0
                            stateActKey = probabilitySolver.stateActionKey(state_idx, action)
                            for nextStateIdx, pNextState in probabilitySolver.stateActionDict[stateActKey].items():
                                curr_exp_value += pNextState*V[nextStateIdx]
                            curr_value =  G[state_idx,action] + curr_exp_value
                            curr_deltaV = np.abs(curr_value - V[state_idx])
                            V[state_idx] = curr_value
                            if curr_deltaV > max_deltaV:
                                max_deltaV = curr_deltaV
                        looping = False
                        if max_deltaV > epsilon:
                            looping = True

                # GT POLICY EVALUATION
                pi_sol = np.load("pi_value_dict.npy")
                ps = probabilitySolver(None,None,None, init=False)
                V_gt = np.zeros(K)
                epsilon = 0
                looping = True
                while looping:
                    max_deltaV = 0
                    for state_idx in range(K):
                        if state_idx != ps.labGemIdx:
                            action = u_opt_ind[state_idx]
                            curr_exp_value = 0.0
                            stateActKey = probabilitySolver.stateActionKey(state_idx, action)
                            for nextStateIdx, pNextState in probabilitySolver.stateActionDict[stateActKey].items():
                                curr_exp_value += pNextState*V_gt[nextStateIdx]
                            curr_value =  G[state_idx,action] + curr_exp_value
                            curr_deltaV = np.abs(curr_value - V_gt[state_idx])
                            V_gt[state_idx] = curr_value
                            if curr_deltaV > max_deltaV:
                                max_deltaV = curr_deltaV
                        looping = False
                        if max_deltaV > epsilon:
                            looping = True

                if np.array_equal(V,V_gt):
                    print("Your policy is CORRECT")
                else:
                    print("Your policy is WRONG")
                print("The maximum value difference is: ", np.linalg.norm(V_gt-V, np.inf))
                
            # Plot results
            print('Plot results')
            MakePlots(map_world, stateSpace, J_opt, u_opt_ind, TERMINAL_STATE_INDEX, "Solution")

    # Terminated
    #PlotCell(map_world, stateSpace[0])
    print('Terminated - close plots to exit program')

    # display graphs
    plt.show()

