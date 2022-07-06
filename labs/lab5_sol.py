import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets

# load the iris dataset from sklearn
def load_iris():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    
    return D, L

# split the dataset into a training set and a test set, 2 to 1 split --> 100 samples to train and 50 to test
# the dataset gets randomized to get the best possible division 
def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)

# reshape to column vector function
def mcol(m):
    return m.reshape((m.size, 1))
        
# reshape to row vector function
def mrow(m):
     return m.reshape((1, m.size))

# function to compute the mean of each feature row of a numpy array
def compute_mean(X):

    empirical_mean=np.array([])

    for i in range(X.shape[0]):
        row = X[i,:]
        mean_row = np.mean(row)
        empirical_mean = np.append(empirical_mean, mean_row)

    return empirical_mean.reshape((empirical_mean.size, 1))

# function to compute the covariance matrix
def compute_cov(X):
    mu = compute_mean(X)
    return np.dot((X-mu), (X-mu).T)/X.shape[1]

def logpdf_1sample(values, Mu, C):

    #computation of the logMultiVariateGaussian for each sample
    P = np.linalg.inv(C)
    logN = -0.5*values.shape[0]*np.log(2*np.pi)-0.5*np.linalg.slogdet(C)[1]-0.5*np.dot(np.dot((values-Mu).T, P), (values-Mu)) 

    return logN.ravel()

def logpdf_GAU_ND(X, mu, C):
    Y = [logpdf_1sample(X[:, i:i+1], mu, C) for i in range(X.shape[1])]
    return np.array(Y).ravel()


if __name__ == '__main__':
    D, L = load_iris()
    # DTR and LTR are training data and labels, DTE and LTE are evaluation data and labels
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    

    # non-logarithmic procedure 
    # mean computation 
    mu_0 = compute_mean(DTR[:, LTR==0])
    mu_1 = compute_mean(DTR[:, LTR==1])
    mu_2 = compute_mean(DTR[:, LTR==2])

    # covariance computation 
    cov_0 = compute_cov(DTR[:, LTR==0])
    cov_1 = compute_cov(DTR[:, LTR==1])
    cov_2 = compute_cov(DTR[:, LTR==2])
    
    # scores computation 
    scores = np.empty([3,50])
    scores[0, :] = mrow(np.exp(logpdf_GAU_ND(DTE, mu_0, cov_0)))
    scores[1, :] = mrow(np.exp(logpdf_GAU_ND(DTE, mu_1, cov_1)))
    scores[2, :] = mrow(np.exp(logpdf_GAU_ND(DTE, mu_2, cov_2)))

    

    # joint scores computation 
    j_scores = np.empty([3,50])
    j_scores[0, :] = scores[0, :]*prior
    j_scores[1, :] = scores[1, :]*prior
    j_scores[2, :] = scores[2, :]*prior
    
    SMarginal = j_scores.sum(0)
    post1 = j_scores / SMarginal
    LPred = post1.argmax(0)
    print(LPred)
    # compare_j_scores = np.load('labs/lab5_results/SJoint_MVG.npy')
    print((LTE==LPred).sum(0)/LTE.shape[0])
