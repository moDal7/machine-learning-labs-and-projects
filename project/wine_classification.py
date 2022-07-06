import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats as ss

def mcol(v):
    return v.reshape((v.size, 1))

def mrow(v):
    return v.reshape((1, v.size))

def load_set(file_name):
    att_list = []
    class_list = []
    with open(file_name) as train:
        for line in train:
            try:
                attrs = line.split(',')[0:11]
                attrs = mcol(np.array([float(i) for i in attrs]))
                wine_class =  int(line.split(',')[-1].strip('\n'))
                att_list.append(attrs)
                np.array(class_list.append(wine_class), dtype=np.int32)
            except:
                pass
         
    return np.hstack(att_list), mrow(np.hstack(class_list))

# function to compute the mean of each feature row of a numpy array
def compute_mean(X):

    empirical_mean=np.array([])

    for i in range(X.shape[0]):
        row = X[i,:]
        mean_row = np.mean(row)
        empirical_mean = np.append(empirical_mean, mean_row)

    return empirical_mean.reshape(empirical_mean.size, 1)

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

def multivariate_gaussian_classifier(DTR, LTR, DTE, LTE):

    DTR_0 = DTR[:, LTR[0, :]==0]
    DTR_1 = DTR[:, LTR[0, :]==1]
    lab = [0,1]
    mu_0 = np.empty([11, 1])
    mu_0 = compute_mean(DTR_0)
    mu_1 = np.empty([11, 1])
    mu_1 = compute_mean(DTR_1)


    cov_0 = compute_cov(DTR_0)
    cov_1 = compute_cov(DTR_1)
    
    scores = np.empty([len(lab), DTE.shape[1]])
    scores[0, :] = mrow(np.exp(logpdf_GAU_ND(DTE, mu_0, cov_0)))
    scores[1, :] = mrow(np.exp(logpdf_GAU_ND(DTE, mu_1, cov_1)))
    

    priors = [0, 0]
    for i in range(len(lab)):
        priors[i] = (LTR[0, :]==i).sum(0)/LTR.shape[1]

    # joint scores computation 
    j_scores = np.empty([len(lab), DTE.shape[1]])
    j_scores[0, :] = scores[0, :]*priors[0]
    j_scores[1, :] = scores[1, :]*priors[1]

    SMarginal = j_scores.sum(0)

    post1 = j_scores / SMarginal
    LPred = post1.argmax(0)
    # compare_j_scores = np.load('labs/lab5_results/SJoint_MVG.npy')
    print("Multivariate Gaussian Classifier accuracy:")
    print((LTE[0, :]==LPred).sum(0)/LTE.shape[1]) 

def gaussianization(data):
    # rank computation 
    rank_matrix = np.empty([data.shape[0], data.shape[1]])

    for i in range(data.shape[0]):
        rank_matrix[i, :] = ss.rankdata(data[i, :])
    
    rank_matrix = (rank_matrix)/(data.shape[1]+2)
    
    # ppf function from scipy.stats module 
    gaussianized_data = norm.ppf(rank_matrix)

    return gaussianized_data



'''
def correlation(DTR):

def exploratory_data_analysis(DTR, LTR):
    mu = compute_mean(DTR)
    cov = compute_cov(DTR)
'''


if __name__ == '__main__':

    train_fname = "./project/data/Train.txt"
    test_fname = "./project/data/Test.txt"
    
    # data and labels loading
    DTR, LTR = load_set(train_fname)
    DTE, LTE = load_set(test_fname)

    # data gaussianization
    DTR_g = gaussianization(DTR)
    DTE_g = gaussianization(DTE)

    # exploratory_data_analysis(DTR, LTR)
    # plot_hist(DTR, LTR)
    # plot_boxplot(DTR, LTR)
    # plot_scatter(DTR, LTR)
    # data_heatmap(DTR, LTR)
    # pca(DTR, LTR)
    # lda(DTR, LTR)
    # compute_dcf(LPred, LTE)
    # feature_gaussianization(DTR, LTR)
    # logistic_regression(DTR, LTR, DTE, LTE)
    # support_vector_machines(DTR, LTR, DTE, LTE)
    # guassian_mixture_models(DTR, LTR, DTE, LTE)

    multivariate_gaussian_classifier(DTR, LTR, DTE, LTE)
    multivariate_gaussian_classifier(DTR_g, LTR, DTE_g, LTE)