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
    # NO
    V = np.zeros(K)
    policy = np.full(K,Constants.STAY)
    ps = probabilitySolver(None,None,None, init=False)
    looping = True
    epsilon = 0.05

    while(looping):
        looping = False
        # update states
        for state_idx in range(K):
            min_value = np.inf

            for action in ps.possibleActionDict[state_idx]:
                curr_exp_value = 0.0
                stateActKey = ps.stateActionKey(state_idx, action)
                for nextStateIdx, pNextState in ps.stateActionDict[stateActKey].items():
                    curr_exp_value += pNextState*V[nextStateIdx]
                curr_value =  G[state_idx,action] + curr_exp_value
                if curr_value < min_value:
                    min_value = curr_value
                    policy[state_idx] = action

            curr_deltaV = np.abs(min_value - V[state_idx])
            V[state_idx] = min_value

            if curr_deltaV > epsilon:
                looping = True

    return V, policy

