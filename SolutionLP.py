import numpy as np
from Constants import *
from ComputeTransitionProbabilities import probabilitySolver
from scipy.optimize import milp, LinearConstraint, Bounds

def idx_vec(position, value, lenght):
    vec = np.zeros(lenght)
    vec[position] = value
    return vec

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

    ps = probabilitySolver(None,None,None, init=False)

    ####################################################
    # FAKE PROBLEM DEFINITION
    # K = 3
    # fakePossActDict = {}
    # fakePossActDict[0] = [0]
    # fakePossActDict[1] = [0]
    # fakePossActDict[2] = [0]
    # fakeStateActDict = {}
    # fakeStateActDict[(0,0)] = [(0,1)]
    # fakeStateActDict[(1,0)] = [(0, 0.5), (1,0.5)]
    # fakeStateActDict[(2,0)] = [(1, 1/3), (2, 2/3)]

    # G[0,0] = 0
    # G[1,0] = 1
    # G[2,0] = 3
    # fakeLabGemsIdx = 0

    #####################################################
    # FIND VALUES WITH LP

    I = [i for i in range(K)]

    c = -np.ones(K)

    constraint_list = []

    # terminal value is set to 0
    A = idx_vec(ps.labGemIdx, 1, K)
    b_u = 0
    b_l = 0
    constraint_list.append(LinearConstraint(A, b_l, b_u))

    # constraints for other values
    for stateIdx in I:
        if stateIdx != ps.labGemIdx:
            for action in ps.possibleActionDict[stateIdx]:
                stateIdxVec = idx_vec(stateIdx, 1, K)
                stateActKey = probabilitySolver.stateActionKey(stateIdx, action)
                pPrimeVec = np.zeros(K)
                for nextStateIdx, pNextState in ps.stateActionDict[stateActKey].items():
                    if nextStateIdx != ps.labGemIdx:
                        pPrimeVec[nextStateIdx] = pNextState
                A = stateIdxVec - pPrimeVec
                b_u = G[stateIdx, action]
                b_l = -np.inf
                constraint_list.append(LinearConstraint(A, b_l, b_u))

    integrality = np.zeros(K)
    result = milp(c=c, constraints=constraint_list, integrality=integrality)
    V = result.x

    #####################################################
    # FIND GREEDY POLICY

    policy = np.zeros(K).astype(int)
    
    for stateIdx in range(K):
        if stateIdx == ps.labGemIdx:
            policy[stateIdx] = Constants.STAY
        else:
            minValue = np.inf
            for action in ps.possibleActionDict[stateIdx]:
                currExpValue = 0.0
                stateActKey = probabilitySolver.stateActionKey(stateIdx, action)
                for nextStateIdx, pNextState in ps.stateActionDict[stateActKey].items():
                    currExpValue += pNextState*V[nextStateIdx]            
                    currValue =  G[stateIdx,action] + currExpValue
                    if currValue < minValue:
                        minValue = currValue
                        policy[stateIdx] = action

    #################################################################
    # GRAPH SEARCH TO FIND IF THE POLICY IS PROPER
    if True:
        print('finding if policy is proper...')
        finalStateIdx = ps.labGemIdx
        proper = True
        minPathProb = 1
        visited = {}
        for firstStateIdx in range(K):
            # print(' ------------ ')
            # print('finding paths from state ' + str(first_state_idx))
            parent = {}
            for idx in range(K):
                visited[idx] = False
            visited[firstStateIdx] = True
            queue = [firstStateIdx]
            founded = False
            while len(queue)!=0 and not founded:
                stateIdx = queue.pop(0)
                if stateIdx == finalStateIdx:
                    founded = True
                    break
                stateActKey = probabilitySolver.stateActionKey(stateIdx, policy[stateIdx])
                for nextStateIdx, _ in ps.stateActionDict[stateActKey].items():
                    if not visited[nextStateIdx]:
                        parent[nextStateIdx] = stateIdx
                        visited[nextStateIdx] = True
                        queue.append(nextStateIdx)
                        
            if not founded:
                print('goal not founded')
                print(firstStateIdx)
                print(policy[firstStateIdx])
                proper = False
            
            # double check
            # print('K is ' + str(K))
            stateIdx = finalStateIdx
            pathProb = 1
            while stateIdx != firstStateIdx:
                # print('stateIdx is ' + str(stateIdx))
                pathProb *= P[parent[stateIdx], stateIdx, policy[parent[stateIdx]]]
                if P[parent[stateIdx], stateIdx, policy[parent[stateIdx]]] <= 0.01:
                    print('error!') 
                    print('parent state: ' + str(parent[stateIdx]))
                    print('to state: ' + str(stateIdx))
                    print('policy: ' + str(policy[stateIdx]))
                    exit()
                stateIdx = parent[stateIdx]
            # print('state ' + str(finalStateIdx) + ' is reachable from state ' + str(firstStateIdx))
            # print('path probability is ' + str(pathProb))
            if minPathProb > pathProb:
                minPathProb = pathProb
        
        if proper:
            print('your policy is proper with min path probability is ' + str(minPathProb))

        else:
            print('your policy is NOT proper.')

    return V, policy

