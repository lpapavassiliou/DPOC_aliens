import numpy as np
import scipy
from Constants import *
from ComputeTransitionProbabilities import probabilitySolver
from concurrent.futures import ProcessPoolExecutor

#import mdptoolbox

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
    solver = probabilitySolver(None, None, None, init=False)

    transition = np.zeros((5,K,K))
    for i in range(K):
        for j in range(K):
            for a in range(5):
                transition[a][i][j] = P[i][j][a]

    print(transition[0])
    value = mdptoolbox.mdp.ValueIteration(transitions=transition, reward=-G, discount=1.0, epsilon=0.0000000000000001)
    print(value.policy)
    return V, policy


