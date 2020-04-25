###############################################################################
# CMPT 727 PA 3
# author: Ya Le, Billy Jun, Xiaocheng Li
# edit: by Heng Liu
# date: Mar 25, 2020
###############################################################################

# Utility code for PA3
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import itertools
from factor_graph import *
from factors import *
from factors import Factor


def loadLDPC(name):
    """
    :param - name: the name of the file containing LDPC matrices

    return values:
    G: generator matrix
    H: parity check matrix
    """
    A = sio.loadmat(name)
    G = A['G']
    H = A['H']
    return G, H


def loadImage(fname, iname):
    '''
    :param - fname: the file name containing the image
    :param - iname: the name of the image
    (We will provide the code using this function, so you don't need to worry too much about it)

    return: image data in matrix form
    '''
    img = sio.loadmat(fname)
    return img[iname]


def applyChannelNoise(y, epsilon):
    '''
    :param y - codeword with 2N entries
    :param epsilon - the probability that each bit is flipped to its complement

    return corrupt message yTilde
    yTilde_i is obtained by flipping y_i with probability epsilon
    '''

    yTilde = np.mod(
        y + np.random.choice([0, 1], size=len(y), p=[1-epsilon, epsilon]
                             ).reshape(y.shape), 2)

    return yTilde


def encodeMessage(x, G):
    '''
    :param - x orginal message
    :param[in] G generator matrix
    :return codeword y=Gx mod 2
    '''
    return np.mod(np.dot(G, x), 2)


def constructFactorGraph(yTilde, H, epsilon):
    '''
    :param - yTilde: observed codeword
        type: numpy.ndarray containing 0's and 1's
        shape: 2N
    :param - H parity check matrix
             type: numpy.ndarray
             shape: N x 2N
    :param epsilon - the probability that each bit is flipped to its complement

    return G factorGraph

    You should consider two kinds of factors:
    - M unary factors
    - N each parity check factors
    '''
    N = H.shape[0]
    M = H.shape[1]
    G = FactorGraph(numVar=M, numFactor=N+M)
    G.var = range(M)
    ##############################################################
    # Q1
    # TODO: your code starts here
    # Add unary factors
    for variable_Y in range(M):
        if(yTilde[variable_Y,0] == 0):
            G.factors.append(Factor(scope = [variable_Y], card = [2], val =np.array([1-epsilon, epsilon]), name="Variable_Y_"+str(variable_Y)))
            Factorfs = variable_Y
            G.factorToVar[Factorfs].append(variable_Y)
            G.varToFactor[variable_Y].append(Factorfs)
        else:
            G.factors.append(Factor(scope = [variable_Y], card = [2], val= np.array([epsilon, 1-epsilon]), name="Variable_Y_"+str(variable_Y)))
            Factorfs = variable_Y
            G.factorToVar[Factorfs].append(variable_Y)
            G.varToFactor[variable_Y].append(Factorfs)
            #Here, the index of factors are the same to index of variables since these are individual factors
        
      
    Factors =[]
    for i in range(N):
        H_p = H[i]
        related_variables = []
        cardinalities=[]
        Factorfs = M + i
        for variable in range(M):
            if(H_p[variable] == 1):
                cardinalities.append(2)
                related_variables.append(variable)
                G.varToFactor[variable].append(Factorfs)        
        #adding These new factors: P
        G.factorToVar[Factorfs] = related_variables
        codeword_size = len(cardinalities)
        values = np.zeros((cardinalities))
        for codeword in itertools.product([0, 1], repeat=codeword_size):
            parity = int(sum(codeword) - 2* int(sum(codeword)/2))
            if(parity == 1):
                values[codeword] = 0.0
            if(parity == 0):
                values[codeword] = 1.0

        #print("asghar")
        #print(values.flat)
        Factors.append(Factor(scope=related_variables, card=cardinalities, val=values, name="P_"+str(i)))


    G.factors = G.factors + Factors


    ##############################################################
    return G


def run_q1():
    yTilde = np.array([[1, 1, 1, 1, 1, 1]]).reshape(6, 1)
    print("yTilde.shape", yTilde.shape)
    H = np.array([
        [0, 1, 1, 0, 1, 0],
        [0, 1, 0, 1, 1, 0],
        [1, 0, 1, 0, 1, 1]])
    epsilon = 0.05
    G = constructFactorGraph(yTilde, H, epsilon)
    ##############################################################
    # Q1
    # TODO: your code starts here
    # Design two invalid codewords ytest1, ytest2 and one valid codewords ytest3.
    #  Report their weights respectively.

    ##############################################################
    ytest1 = np.array([0, 1, 1, 0, 1, 0])
    ytest2 = np.array([1, 0, 1, 1, 0, 1])
    ytest3 = np.array([1, 0, 1, 1, 1, 1])
    print(
        G.evaluateWeight(ytest1),
        G.evaluateWeight(ytest2),
        G.evaluateWeight(ytest3))


def run_q3():
    '''
    In part b, we provide you an all-zero initialization of message x, you should
    apply noise on y to get yTilde, and then do loopy BP to obatin the
    marginal probabilities of the unobserved y_i's.
    '''
    G, H = loadLDPC('ldpc36-128.mat')

    print(H)
    print(H.shape)
    epsilon = 0.05
    N = G.shape[1]
    x = np.zeros((N, 1), dtype='int32')
    y = encodeMessage(x, G)

    yTilde = applyChannelNoise(y, epsilon)
    G = constructFactorGraph(yTilde, H, epsilon)
    values = []
    G.runParallelLoopyBP(30)
    for var in G.var:
        values.append(G.estimateMarginalProbability(var)[1])
    plt.figure()
    plt.title("Plot of the estimated posterior probability P(Yi=1|Y~)")
    plt.ylabel("Probability of Bit Being 1")
    plt.xlabel("Bit Index of Received Message")
    plt.plot(range(len(G.var)), values)
    plt.ylim((-0.00001,5e-30))
    plt.savefig('q3', bbox_inches='tight')
    plt.show()
    plt.close()

    # Verify we get a valid codeword.
    MMAP = G.getMarginalMAP()
    
    return G.evaluateWeight(MMAP)



def run_q4(numTrials, error, iterations=30):
    '''
    param - numTrials: how many trials we repreat the experiments
    param - error: the transmission error probability
    param - iterations: number of Loopy BP iterations we run for each trial
    '''
    G, H = loadLDPC('ldpc36-128.mat')
    N = G.shape[1]
    x = np.zeros((N, 1), dtype='int32')
    y = encodeMessage(x, G)
    plt.figure()
    plt.title("Plot of the Hamming Distance Between MMAP and True Over 8 "
              "Trials")
    plt.ylabel("Hamming Distance")
    plt.xlabel("Iteration Number of Loopy Belief Propagation")
    for trial in range(numTrials):
        yTilde = applyChannelNoise(y, error)
        G = constructFactorGraph(yTilde, H, error)
        values = []
        for it in range(iterations):
            G.runParallelLoopyBP(1)
            if it % 10 == 0 and it > 0:
                print("Finished iteration %s of Loopy" % it)
            MMAP = G.getMarginalMAP()
            hamming_distance = np.sum(MMAP)
            values.append(hamming_distance)
        plt.plot(values)

    plt.savefig('q4_epsilon=' + str(int(100*error)), bbox_inches='tight')
    plt.show()
    plt.close()






################################################################

if __name__ == "__main__":
    '''
    TODO modify or use run_q1(), run_q3(), and run_q4() to
    run your implementation for this assignment.

    print('Doing Q1: Should see 0.0, 0.0, >0.0')
    run_q1()
    print('Doing Q3 ')
    print(run_q3())
    print('Doing Q4')
    run_q4(8, 0.06)
    '''
    print("STARTTTTTTTTTT")
    run_q3()
    run_q4(8, 0.06)
