import math
try:
    import numpy
except:
    print("This implementation requires the numpy module.")
    exit(0)

###############################################################################

"""
@INPUT:
    R     : a matrix to be factorized, dimension N x M
    P     : an initial matrix of dimension N x K
    Q     : an initial matrix of dimension M x K
    K     : the number of latent features
    steps : the maximum number of steps to perform the optimisation
    alpha : the learning rate
    beta  : the regularization parameter
@OUTPUT:
    the final matrices P and Q
"""

def globalAverage(R):
    sum = 0
    n = 0
    for i in range(len(R)):
        for j in range(len(R[i])):
            if R[i][j] > 0:
                sum += R[i][j] 
                n += 1
    return (sum/n)

def userBias(R,miu):
    user_bias = []
    for i in range(len(R)):
        sum = 0
        n = 0
        for j in range(len(R[i])):
            if (R[i][j] > 0):
                sum += R[i][j]
                n += 1
        user_bias.append((sum/n) - miu)
    return user_bias

def itemBias(R,miu):
    item_bias = [0 for x in range(len(R[0]))]
    sum = [0 for x in range(len(R[0]))]
    n = [0 for x in range(len(R[0]))]
    for i in range(len(R)):
        for j in range(len(R[i])):
            if (R[i][j] > 0):
                sum[j] += R[i][j]
                n[j] += 1
    for i in range(len(sum)):
        item_bias[i] = (sum[i]/n[i]) - miu
    return item_bias


def matrix_factorization(R, P, Q, K, miu, u_Bias, i_Bias, steps=100, alpha=0.01, beta=0.02):
    Q = Q.T
    for i in range(len(R)):
        for j in range(len(R[i])):
            if R[i][j] > 0:
                R[i][j] += (miu + u_Bias[i] + i_Bias[j])

    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - miu - u_Bias[i] - i_Bias[j] -numpy.dot(P[i,:],Q[:,j]) #
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = numpy.dot(P,Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        if e < 0.001:
            break
    return P, Q.T

def mse(target,prediction):
    mse = 0
    n = 0
    for i in range(len(prediction)):
            for j in range(len(prediction[i])):
                if target[i][j] > 0:
                    mse += pow(target[i][j] - prediction[i][j], 2)
                    n += 1
    return (mse/n)

###############################################################################

if __name__ == "__main__":
    R = [
          [0,0,3,0,0,1,0,0,0,0,4,0,5,0,5,0,0,0,0,0],
          [4,3,0,2,0,2,3,0,0,0,0,0,1,3,0,4,0,0,0,0],
          [0,5,3,1,0,1,0,0,1,0,5,4,0,0,5,0,0,0,0,0],
          [5,0,0,1,3,2,0,4,0,3,4,0,2,3,0,0,0,0,0,0],
          [2,0,3,0,0,3,1,3,1,0,0,0,4,0,0,0,0,0,0,0],
          [0,0,2,0,0,0,0,0,1,0,0,0,0,3,4,4,0,0,0,0],
          [5,0,4,2,0,1,0,5,0,0,0,0,0,0,0,0,0,0,5,0],
          [5,0,4,0,0,1,0,4,0,3,0,0,3,0,3,5,0,0,0,0],
          [0,0,0,0,0,0,4,0,0,0,0,5,0,2,5,0,2,0,0,0],
          [3,0,4,0,0,2,0,0,1,0,2,0,4,0,0,0,0,0,0,1],
          [2,0,5,0,0,0,2,0,0,0,0,3,0,0,5,0,0,0,0,2],
          [4,5,0,0,0,0,0,0,1,0,0,0,3,4,5,4,0,0,0,0],
          [5,0,0,0,4,3,5,0,0,2,0,4,0,5,0,0,0,4,0,2],
          [0,0,3,0,0,0,0,4,5,0,0,0,5,0,5,0,0,0,0,0],
          [4,0,0,2,0,3,5,0,0,0,0,0,0,0,0,0,0,0,0,0],
          [5,0,3,0,0,0,4,0,2,0,0,0,4,0,5,0,0,3,0,0],
          [1,0,4,3,0,0,2,2,0,1,0,0,5,4,0,0,0,0,0,1],
          [3,0,4,2,4,2,3,0,0,0,0,0,0,0,0,0,0,0,0,0],
          [4,0,3,0,0,1,4,5,0,3,5,0,0,0,4,0,0,0,0,0],
          [0,1,0,1,0,2,0,3,0,1,0,5,0,0,5,0,0,0,1,0],
          [0,0,5,2,5,1,3,0,0,0,1,4,0,0,1,0,0,0,0,0],
          [0,0,0,0,1,0,4,0,1,2,5,0,0,0,5,0,3,0,2,2],
          [0,0,5,0,4,0,0,0,0,0,3,5,0,5,0,0,0,0,0,0],
          [0,3,0,0,4,0,0,3,0,1,0,0,0,4,0,0,0,1,0,1],
          [3,5,4,1,0,0,0,0,0,0,3,0,0,0,0,0,0,0,2,2],
          [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,2,0,2,3],
          [0,0,0,0,2,0,0,0,0,0,0,0,3,0,0,0,0,3,2,2],
        ]

    R = numpy.array(R)

    N = len(R)
    M = len(R[0])
    K = 5

    P = numpy.random.rand(N,K)
    Q = numpy.random.rand(M,K)

    print("the Original Matrix")
    print(R)
    
    miu = globalAverage(R)
    print('\nGlobal Average:', miu)

    u_Bias = userBias(R,miu)
    i_Bias = itemBias(R,miu)
    print('\nUser bias:', u_Bias)
    print('\nItem bias:', i_Bias)

    nP, nQ = matrix_factorization(R, P, Q, K,miu, u_Bias, i_Bias)
    print("\nThe Approximation matrix by MF")
    prediction = numpy.dot(nP, nQ.T)
    print(prediction)
    
    target = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
       [0,0,3,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
       [5,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
       [0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
       [0,0,0,3,3,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
       [5,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0,0,0,4,0,0,0,5,0,0,0,0,0],
       [0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0],
       [3,0,5,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
       [0,0,0,0,4,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0,0,0,0,0,0,5,0,0,0,0,0,0],
       [0,0,0,0,0,2,0,0,0,3,0,0,0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
       [5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
       [0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
       [0,0,0,0,2,1,0,0,0,0,4,0,0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0],
       [0,0,5,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0,0,1,0,0,4,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0],
       [4,0,3,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,5,1,0,0,0,0,0,4,0,0,0,0,0],
       [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
       [0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,3,0,0,0,0]]
    
    mse = mse(target,prediction)
    print("\nMSE = ", mse)
    print("\nRMSE = ", math.sqrt(mse))

