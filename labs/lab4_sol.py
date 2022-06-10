import numpy as np
import matplotlib.pyplot as plt


def createVariables():
    # create variables to test the functions
    X = np.linspace(-8, 12, 1000)
    # make it a row matrix
    x = vrow(X)
    m = np.ones((1,1)) * 1.0
    C = np.ones((1,1)) * 2.0

    return x, m, C

def vrow(m):
     return m.reshape((1, m.size))

def mcol(m):
    return m.reshape((m.size, 1))

def logpdf_1sample(values, Mu, C):

    #computation of the logMultiVariateGaussian for each sample
    P = np.linalg.inv(C)
    logN = -0.5*values.shape[0]*np.log(2*np.pi)-0.5*np.linalg.slogdet(C)[1]-0.5*np.dot(np.dot((values-Mu).T, P), (values-Mu)) 

    return logN.ravel()

def logpdf_GAU_ND(X, mu, C):
    Y = [logpdf_1sample(X[:, i:i+1], mu, C) for i in range(X.shape[1])]
    return np.array(Y).ravel()


def plot_Multivariate(MVG, X):
    arrayx = np.array(X)
    plt.figure()
    plt.plot(arrayx.ravel(), np.exp(MVG))
    plt.show()



if __name__ == '__main__': 
    
    x, mu, c = createVariables()
    log_MVG = logpdf_GAU_ND(x, mu, c)
    plot_Multivariate(log_MVG, x)


