import numpy as np
import scipy
from Constants import *
from ComputeTransitionProbabilities import probabilitySolver

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
    
    # #Do you need to do something with the terminal state before solving the problem?


################################################################
    # POLICY ITERATION -SIMONE

    # V = np.zeros(K).astype(float)
    # pi = np.full(K, Constants.STAY).astype(int)
    
    # gamma = 1
    # epsilon = 0.01

    # solver = probabilitySolver(None, None, None, init=False)


    # policy_diff = True
    # while policy_diff:
    #     policy_diff = False

    #     #print("New policy evaluation")
    #     # ------------------------------------------------------------
    #     value_diff = True
    #     while value_diff:
    #         value_diff = False
            
    #         # for each state in K
    #         for stateIdx in range(0, K):
    #             action = pi[stateIdx][0]

    #             sumNextStatesCost = 0
    #             stateActKey = probabilitySolver.stateActionKey(stateIdx, action)
    #             for nextStateIdx, pNextState in solver.stateActionDict[stateActKey].items():
    #                 sumNextStatesCost += pNextState*V[nextStateIdx]

    #             V_prime = G[stateIdx][action] + gamma * sumNextStatesCost

    #             if abs(V[stateIdx] - V_prime) > epsilon:
    #                 value_diff = True

    #             V[stateIdx] = V_prime
        
    #     #print("New policy improvement")
    #     #----------------------------------------------------------------------------
    #     # improve policy
    #     pi_new = np.zeros_like(pi).astype(int)

    #     for stateIdx in range(K):
    #         min_action_value = np.inf
    #         min_action = -1

    #         for action in solver.getPossibleActions(solver.stateSpace[stateIdx]):
    #             sumNextStatesCost = 0
    #             stateActKey = probabilitySolver.stateActionKey(stateIdx, action)
    #             for nextStateIdx, pNextState in solver.stateActionDict[stateActKey].items():
    #                 sumNextStatesCost += pNextState*V[nextStateIdx]

    #             action_value = G[stateIdx][action] + gamma * sumNextStatesCost

    #             if min_action_value > action_value:
    #                 min_action_value = action_value
    #                 min_action  = action

    #         pi_new[stateIdx] = min_action

    #         if pi_new[stateIdx] != pi[stateIdx]:
    #                 policy_diff = True

    #     pi = pi_new

    # return V, pi




#########################################################################
# GRAPH SEARCH TO FIND IF A POLICY IS PROPER

    ps = probabilitySolver(None,None,None, init=False)
    final_state_idx = ps.labGemIdx

    visited = {}
    for first_state_idx in range(K):
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
            stateActKey = probabilitySolver.stateActionKey(state_idx, Constants.STAY)
            for next_state_idx, _ in probabilitySolver.stateActionDict[stateActKey].items():
                if not visited[next_state_idx]:
                    visited[next_state_idx] = True
                    queue.append(next_state_idx)
        if not founded:
            print('goal not founded')
    return None,None





# ######################################
# # POLICY ITERATION - DOLU
# # with Value Iteration initial guess


#     V = np.zeros(K)
#     policy = np.full(K,Constants.STAY)
#     gamma = 1
#     ps = probabilitySolver(None,None,None, init=False)
#     looping = True
#     epsilon = 0.000000000000000000000001

#     while(looping):

#         max_deltaV = 0

#         # update states
#         for state_idx in range(K):

#             min_value = np.inf
#             for action in ps.getPossibleActions(ps.stateSpace[state_idx]):
#                 curr_exp_value = 0.0
#                 stateActKey = probabilitySolver.stateActionKey(state_idx, action)
#                 for nextStateIdx, pNextState in probabilitySolver.stateActionDict[stateActKey].items():
#                     curr_exp_value += pNextState*V[nextStateIdx]
#                 curr_value =  G[state_idx,action] + gamma*curr_exp_value
#                 if curr_value < min_value:
#                     min_value = curr_value
#                     policy[state_idx] = action
            
#             old_V_state = V[state_idx]
#             V[state_idx] = min_value
#             curr_deltaV = np.abs(old_V_state - V[state_idx])

#             if curr_deltaV > max_deltaV:
#                 max_deltaV = curr_deltaV


#         # checks for another iteration
#         looping = False
#         if max_deltaV > epsilon:
#             looping = True


#     ###
#     V = np.zeros(K)
#     gamma = 1
#     ps = probabilitySolver(None,None,None, init=False)
#     changing = True
#     # epsilon = 0.000001

#     while changing:

#         # ---------------------------------------------------------------------
#         # POLICY EVALUATION

#         looping = True
#         while looping:

#             max_deltaV = 0

#             # update states
#             for state_idx in range(K):

#                 if state_idx != ps.labGemIdx:

#                     min_value = np.inf
#                     action = policy[state_idx]
#                     curr_exp_value = 0.0
#                     stateActKey = probabilitySolver.stateActionKey(state_idx, action)
#                     for nextStateIdx, pNextState in probabilitySolver.stateActionDict[stateActKey].items():
#                         curr_exp_value += pNextState*V[nextStateIdx]
#                     curr_value =  G[state_idx,action] + gamma*curr_exp_value
#                     if curr_value < min_value:
#                         min_value = curr_value
#                         policy[state_idx] = action
                    
#                     old_V_state = V[state_idx]
#                     V[state_idx] = min_value
#                     curr_deltaV = np.abs(old_V_state - V[state_idx])

#                     if curr_deltaV > max_deltaV:
#                         max_deltaV = curr_deltaV


#                 # checks for another iteration
#                 looping = False
#                 if max_deltaV > epsilon:
#                     looping = True
            

#         # ---------------------------------------------------------------------
#         # POLICY IMPROVEMENT

#         for state_idx in range(K):
#             minimum = np.inf
#             for action in ps.getPossibleActions(ps.stateSpace[state_idx]):
#                 curr_exp_value = 0
#                 stateActKey = probabilitySolver.stateActionKey(state_idx, action)
#                 for nextStateIdx, pNextState in probabilitySolver.stateActionDict[stateActKey].items():
#                     curr_exp_value += pNextState*V[nextStateIdx]
#                 curr_value =  G[state_idx,action] + gamma*curr_exp_value

#                 if curr_exp_value < minimum:
#                     minimum = curr_exp_value
#                     best_action = action

#             changing = False
#             if policy[state_idx] != best_action:
#                 changing = True
#                 policy[state_idx] = best_action
#                 print('\nfounded improvement for state:')
#                 print(state_idx)


#     # return V, policy









# ####################################################################################
# # VALUE ITERATION DOLUSH

#     V = np.zeros(K)
#     policy = np.full(K,Constants.STAY)
#     gamma = 1
#     ps = probabilitySolver(None,None,None, init=False)
#     looping = True
#     epsilon = 0

#     while(looping):

#         max_deltaV = 0

#         # update states
#         for state_idx in range(K):

#             min_value = np.inf
#             for action in ps.getPossibleActions(ps.stateSpace[state_idx]):
#                 curr_exp_value = 0.0
#                 stateActKey = probabilitySolver.stateActionKey(state_idx, action)
#                 for nextStateIdx, pNextState in probabilitySolver.stateActionDict[stateActKey].items():
#                     curr_exp_value += pNextState*V[nextStateIdx]
#                 curr_value =  G[state_idx,action] + gamma*curr_exp_value
#                 if curr_value < min_value:
#                     min_value = curr_value
#                     policy[state_idx] = action
            
#             old_V_state = V[state_idx]
#             V[state_idx] = min_value
#             curr_deltaV = np.abs(old_V_state - V[state_idx])

#             if curr_deltaV > max_deltaV:
#                 max_deltaV = curr_deltaV


#         # checks for another iteration
#         looping = False
#         if max_deltaV > epsilon:
#             looping = True

#     return V, policy

