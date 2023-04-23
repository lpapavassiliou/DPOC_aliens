import numpy as np
import scipy
from Constants import *
from ComputeTransitionProbabilities import probabilitySolver
from concurrent.futures import ProcessPoolExecutor

import random
from MakePlots import *
#from profilehooks import profile, coverage

def Solution(P, G, K, TERMINAL_STATE_INDEX):

    """
    Solve the stochastic shortest path problem by Value Iteration, Policy iteration or Linear programming
    
    Computes the optimal cost and
    the optimal control input for each state of the state space.
    @param P: A (K x K x L)-matrix containing the transition probabilities
                          between all states in the state space for all control inputs.
                          The entry P(i, j, l) represents the transition probability
                          from state i to state j if control input l is applied.
    @param G: A (K x L)-matrix containing the stage costs of all states in
                          the state space for all control inputs. The entry G(i, l)
                          represents the cost if we are in state i and apply control
                          input l.
    @param K: An integer representing the total number of states in the state space
    @param TERMINAL_STATE_INDEX: An integer representing the index of the terminal state in the state space

    @return J_opt:
                A (K x 1)-matrix containing the optimal cost-to-go for each
                element of the state space.

    @return u_opt_ind:
                A (K x 1)-matrix containing the index of the optimal control
                input for each element of the state space. Mapping of the
                terminal state is arbitrary (for example: STAY).

    """
    
    #Do you need to do something with the terminal state before solving the problem?
    
    ps = probabilitySolver(None,None,None, init=False)

    #################################################################
    # FAKE PROBLEM DEFINITION

    # K = 4
    # fake_last_state = 3
    # fake_dict = {}
    # fake_dict[0] = [0,1]
    # fake_dict[1] = [0]
    # fake_dict[2] = [3]

    #################################################################
    # DEFINE A POLICY

    policy = np.full(K,Constants.STAY)
    for i in [15, 32, 70]:
        policy[i] = Constants.NORTH
    for i in [13, 28, 60]:
        policy[i] = Constants.SOUTH
    for i in [23, 5, 7]:
        policy[i] = Constants.EAST

    #################################################################
    # GRAPH SEARCH TO FIND IF THE POLICY IS FEASIBLE

    for state in range(K):
        if policy[state] not in ps.possibleActionDict[state]:
            print('action not possible for state ' + str(state))
    print('your policy is feasible.')


    #################################################################
    # GRAPH SEARCH TO FIND IF THE POLICY IS PROPER

    print('finding if policy is proper...')
    final_state_idx = ps.labGemIdx
    proper = True
    min_path_prob = 1
    visited = {}
    for first_state_idx in range(K):
        # print(' ------------ ')
        # print('finding paths from state ' + str(first_state_idx))
        parent = {}
        for idx in range(K):
            visited[idx] = False
        visited[first_state_idx] = True
        queue = [first_state_idx]
        founded = False
        while len(queue)!=0 and not founded:
            state_idx = queue.pop(0)
            if state_idx == final_state_idx:
                founded = True
                break
            stateActKey = probabilitySolver.stateActionKey(state_idx, policy[state_idx])
            for next_state_idx, _ in ps.stateActionDict[stateActKey].items():
                if not visited[next_state_idx]:
                    parent[next_state_idx] = state_idx
                    visited[next_state_idx] = True
                    queue.append(next_state_idx)
                    
        if not founded:
            print('goal not founded')
            proper = False
        
        # double check
        # print('K is ' + str(K))
        state_idx = final_state_idx
        path_prob = 1
        while state_idx != first_state_idx:
            # print('state_idx is ' + str(state_idx))
            path_prob *= P[parent[state_idx], state_idx, policy[parent[state_idx]]]
            if P[parent[state_idx], state_idx, policy[parent[state_idx]]] <= 0.01:
                print('error!') 
                print('parent state: ' + str(parent[state_idx]))
                print('to state: ' + str(state_idx))
                print('policy: ' + str(policy[state_idx]))
                print(P[9, 13, Constants.SOUTH])
                exit()
            state_idx = parent[state_idx]
        # print('state ' + str(final_state_idx) + ' is reachable from state ' + str(first_state_idx))
        # print('path probability is ' + str(path_prob))
        if min_path_prob > path_prob:
            min_path_prob = path_prob
        
    if proper:
        print('your policy is proper with min path probability is ' + str(min_path_prob))

    else:
        print('your policy is NOT proper.')


    ##############################################################
        #  POLICY ITERATION (Ludo)
    print('policy iteration running...')
    V = np.zeros(K)
    changing = True
    epsilon = 0

    while changing:

        # ---------------------------------------
        # POLICY EVALUATION

        looping = True
        while looping:

            max_deltaV = 0

            # update states
            for state_idx in range(K):

                if state_idx != ps.labGemIdx:

                    action = policy[state_idx]
                    curr_exp_value = 0.0
                    stateActKey = probabilitySolver.stateActionKey(state_idx, action)
                    for nextStateIdx, pNextState in probabilitySolver.stateActionDict[stateActKey].items():
                        curr_exp_value += pNextState*V[nextStateIdx]
                    curr_value =  G[state_idx,action] + curr_exp_value
                    
                    curr_deltaV = np.abs(curr_value - V[state_idx])
                    V[state_idx] = curr_value

                    if curr_deltaV > max_deltaV:
                        max_deltaV = curr_deltaV


                # checks for another iteration
                looping = False
                if max_deltaV > epsilon:
                    looping = True

        # -------------------------------------------
        # POLICY IMPROVEMENT
        print('improving policy...')

        changing = False
        for state_idx in range(K):

            minimum = np.inf
            for action in ps.possibleActionDict[state_idx]:

                curr_exp_value = 0
                stateActKey = probabilitySolver.stateActionKey(state_idx, action)
                for nextStateIdx, pNextState in probabilitySolver.stateActionDict[stateActKey].items():
                    curr_exp_value += pNextState*V[nextStateIdx]
                curr_value =  G[state_idx,action] + curr_exp_value

                if curr_value < minimum:
                    minimum = curr_value
                    best_action = action

            if policy[state_idx] != best_action:
                changing = True
                policy[state_idx] = best_action
            


