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



    # 0: ludo
    # 1: simo
    VI_simo = 0
    PI_simo = 0

    # VI epsilon
    epsilon = 0.5

    # PI epsilon
    pi_episilon = 0.05


    # STATE - POSSIBLE ACTIONS DICTIONARY

    ps = probabilitySolver(None,None,None, init=False)

    if not VI_simo:
        ##############################################################
        #  1. VALUE ITERATION (Ludo)
        #     set with epsilon = 1

        V = np.zeros(K)
        policy = np.full(K,Constants.STAY)
        looping = True

        while(looping):
            looping = False
            # update states
            for state_idx in range(K):
                if state_idx != K:
                    min_value = np.inf
                    for action in ps.possibleActionDict[state_idx]:
                        curr_exp_value = 0.0
                        stateActKey = probabilitySolver.stateActionKey(state_idx, action)
                        for nextStateIdx, pNextState in probabilitySolver.stateActionDict[stateActKey].items():
                            curr_exp_value += pNextState*V[nextStateIdx]
                        curr_value =  G[state_idx,action] + curr_exp_value
                        if curr_value < min_value:
                            min_value = curr_value
                            policy[state_idx] = action
                                    
                    curr_deltaV = np.abs(min_value - V[state_idx])
                    V[state_idx] = min_value
                    if curr_deltaV > epsilon:
                        looping = True


    if VI_simo:
    ##############################################################
    #  1. VALUE ITERATION (Simo)

        V = np.zeros(K)
        policy = np.full(K,Constants.STAY)
        looping = True

        while(looping):
            looping = False
            # update states
            a = 0
            for state_idx in range(K):
                if state_idx != K:
                    min_value = np.inf
                    for action in ps.possibleActionDict[state_idx]:
                        curr_exp_value = 0.0
                        stateActKey = probabilitySolver.stateActionKey(state_idx, action)
                        for nextStateIdx, pNextState in probabilitySolver.stateActionDict[stateActKey].items():
                            curr_exp_value += pNextState*V[nextStateIdx]
                        curr_value =  G[state_idx,action] + curr_exp_value
                        if curr_value < min_value:
                            min_value = curr_value
                            policy[state_idx] = action
                                    
                    curr_deltaV = np.abs(min_value - V[state_idx])
                    V[state_idx] = min_value
                    if curr_deltaV > epsilon:
                        looping = True


#------------------------------------
    # SAVE VI POLICY

    V_vi = V
    policy_vi = policy.copy()
    epsilon = pi_episilon
#------------------------------------




    if not PI_simo:
    ##############################################################
    #  2. POLICY ITERATION (Ludo)

        V = np.zeros(K)
        changing = True

        while changing:

            # ---------------------------------------------------------------------
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

            # ---------------------------------------------------------------------
            # POLICY IMPROVEMENT

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



    if PI_simo:
    ##############################################################
    #  2. POLICY ITERATION (Simo)

        V = np.zeros(K)
        policy_diff = True

        while policy_diff:
            
            policy_diff = False


            # policy evaluation

            looping = True
            while(looping):
                looping = False
                for state_idx in range(K):
                    
                    action = policy[state_idx]
                    curr_exp_value = 0.0
                    stateActKey = probabilitySolver.stateActionKey(state_idx, action)
                    for nextStateIdx, pNextState in probabilitySolver.stateActionDict[stateActKey].items():
                        curr_exp_value += pNextState*V[nextStateIdx]
                    curr_value =  G[state_idx,action] + curr_exp_value
                    curr_deltaV = np.abs(curr_value - V[state_idx])
                    V[state_idx] = curr_value
                    if curr_deltaV > epsilon:
                        looping = True
                        
                
            # improve policy
            pi_new = np.zeros_like(policy).astype(int)

            for stateIdx in range(0, K):

                min_action_value = np.inf
                min_action = -1

                for action in ps.possibleActionDict[state_idx]:
                    sumNextStatesCost = 0
                    stateActKey = probabilitySolver.stateActionKey(stateIdx, action)
                    for nextStateIdx, pNextState in ps.stateActionDict[stateActKey].items():
                        sumNextStatesCost += pNextState*V[nextStateIdx]

                    action_value = G[stateIdx][action] + sumNextStatesCost

                    if min_action_value > action_value:
                        min_action_value = action_value
                        min_action  = action

                pi_new[stateIdx] = min_action

                if pi_new[stateIdx] != policy[stateIdx]:
                        policy_diff = True

            policy = pi_new



    #####
    # CONSLUSION

    # different = False
    # for state_idx in range(K):
    #     if policy_vi[state_idx] != policy[state_idx]:
    #         different = True
    #         break
    # if not different:
    #     print('\nsame policy founded!!!!\n')
    # else:
    #     print('\nThe two polices are different')


    return V, policy