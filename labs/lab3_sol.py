# libraries import
import os
import matplotlib.pyplot as plt
import sklearn.datasets
import numpy as np

# load function, iris dataset from sklearn library
def load():
    return sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']

# reshape to column vector function
def mcol(m):
    return m.reshape((m.size, 1))
        
# reshape to row vector function
def mrow(m):
     return m.reshape((1, m.size))

# function to compute the mean of each column of a numpy array
def compute_mean(X):
    return mcol(X.mean(1))

# function to compute the covariance matrix
def compute_cov(X):
    mu = compute_mean(X)
    return np.dot((X-mu), (X-mu).T)/X.shape[1]

def pca(iris_data, targets):
    

    # mean computation for data centering, note that it will be a 1-D row array
    mu = compute_mean(iris_data)
    mu = mcol(mu)

    covar = compute_cov(iris_data)

    # computation of the eigenvalues/eigenvectors: 
    # s contains the eigenvalues from smallest to largest
    # U contains the eigenvectors as columns
    s, U = np.linalg.eigh(covar)

    # choose the number of eigenvectors you are going to use
    m = 2
    # take the principal components
    pc = U[:, [-m+1,-m]]
    


    # projecting the original datasets along these principal components
    DProjList = [] 

    for i in range(iris_data.shape[1]):
        xi = mcol(iris_data[:, i])
        yi = np.dot(pc.T, xi)
        DProjList.append(yi)

    # transformation into the new transform data matrix
    DY = np.hstack(DProjList)

    return DY, targets

def pcaPlot(D, L):

    D0 = D[:, L==0]
    D1 = D[:, L==1]
    D2 = D[:, L==2]

    plt.figure()
    plt.xlabel("First principal component")
    plt.ylabel("Second principal component")

    plt.scatter(D0[0, :], D0[1, :], label = 'Setosa')
    plt.scatter(D1[0, :], D1[1, :], label = 'Versicolor')
    plt.scatter(D2[0, :], D2[1, :], label = 'Virginica')
        
    plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
    plt.savefig('labs/pcaScatter.pdf')
    plt.show()

def lda(iris_data, targets):

    # computation of the within class covariance matrix
    SW = []
    for i in [0, 1, 2]:
        SW += (targets==i).sum() * compute_cov(iris_data[:, targets==i])

    SB = []
    muG = compute_mean(iris_data)
    for i in [0, 1, 2]:
        D = iris_data[:, targets==i]
        mu = compute_mean(D)

        SB += D.shape[1] * np.dot((mu - muG), (mu - muG).T)
    return SB / iris_data.shape[1]

    # missing parts - lesson about lab 3 around from min 40

def ldaPlot(D, L):

    plt.figure()
    plt.xlabel("First direction")
    plt.ylabel("Second direction")

    plt.scatter(D0[0, :], D0[1, :], label = 'Setosa')
    plt.scatter(D1[0, :], D1[1, :], label = 'Versicolor')
    plt.scatter(D2[0, :], D2[1, :], label = 'Virginica')
        
    plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
    plt.savefig('labs/pcaScatter.pdf')
    plt.show()


# executing all the functions
if __name__ == '__main__':
    
    # load the dataset and the targets
    iris_data, targets = load()
    
    # pca analysis
    DY, targets = pca(iris_data, targets)    
    pcaPlot(DY, targets)

    # lda analysis
    ldaMatrix = lda(iris_data, targets)
    ldaPlot(ldaMatrix, targets)