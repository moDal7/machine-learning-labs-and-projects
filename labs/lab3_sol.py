# libraries import
import os
import matplotlib.pyplot as plt
import sklearn.datasets
import numpy as np

# load function, iris dataset from sklearn library
def load():
    return sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']

def pca(iris_data, targets):
    
    # reshape to column vector function
    def mcol(m):
        return m.reshape((m.size, 1))
        
    # reshape to row vector function
    def mrow(m):
        return m.reshape((1, m.size))


    # mean computation for data centering, note that it will be a 1-D row array
    mu = iris_data.mean(1)
    mu = mcol(mu)

    #centering of the data
    iris_centered = iris_data - mu
    N = iris_centered.shape[1]

    covar = (np.dot(iris_centered, iris_centered.T))/N

    # computation of the eigenvalues/eigenvectors: 
    # s contains the eigenvalues from smallest to largest
    # U contains the eigenvectors as columns
    s, U = np.linalg.eigh(covar)

    # choose the number of eigenvectors you are going to use
    m = 2
    print(U)
    # take the principal components
    pc = U[:, [-m+1,-m]]
    print(pc)


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

# executing all the functions
if __name__ == '__main__':
    
    iris_data, targets = load()
    DY, targets = pca(iris_data, targets)    
    pcaPlot(DY, targets)