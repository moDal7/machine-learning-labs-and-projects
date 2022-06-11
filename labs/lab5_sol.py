import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets

# load the iris dataset from sklearn
def load_iris():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()
    ['target']
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


if __name__ == '__main__':
    D, L = load_iris()
    # DTR and LTR are training data and labels, DTE and LTE are evaluation data and labels
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)