from cmath import log
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import norm
import scipy.stats as ss
import scipy.optimize as so
import scipy.special as sp
from sklearn.model_selection import KFold


# files loaded into numpy arrays
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

# mcol function: reshapes the array into a column array
def mcol(v):
    return v.reshape((v.size, 1))

# mrow function: reshapes the array into a row array
def mrow(v):
    return v.reshape((1, v.size))

# function to compute the mean of each feature of a numpy array
def compute_mean(X):

    empirical_mean=np.array([])

    for i in range(X.shape[0]):
        row = X[i,:]
        mean_row = np.mean(row)
        empirical_mean = np.append(empirical_mean, mean_row)

    return empirical_mean.reshape(empirical_mean.size, 1)

# function to compute the covariance matrix of the dataset
def compute_cov(X):
    mu = compute_mean(X)
    return np.dot((X-mu), (X-mu).T)/X.shape[1]

def compute_variance(X):

    variance=np.array([])

    for i in range(X.shape[0]):
        row = X[i,:]
        variance_row = np.var(row)
        variance = np.append(variance, variance_row)

    return variance.reshape(variance.size, 1)

# function to compute the correlation matrix of the dataset, and to plot the correlation heatmap
def correlation_matrix(data, feature_names):
    correlation = np.corrcoef(data)
    correlation_around = np.around(correlation, 1)
    fig, ax = plt.subplots()
    im = ax.imshow(correlation_around, cmap='coolwarm')

    # show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(feature_names)), labels=feature_names)
    ax.set_yticks(np.arange(len(feature_names)), labels=feature_names)

    # rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            text = ax.text(j, i, correlation_around[i, j],
                        ha="center", va="center", color="w", size=8)

    ax.set_title("Correlation Coefficients")
    fig.tight_layout()
    plt.show()  
'''
def kfold_split(D, L, k, seed=0):
    nTest = int(D.shape[1]*(1/k))
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return DTR, LTR, DTE, LTE
'''

# general exploratory data analysis of the whole dataset, featuring means, covariance matrix and skewness of the data
# prints out the data on the terminal
def eda(data, labels, feature_names):
    mean = compute_mean(data)
    mean_0 = compute_mean(data[:, labels[0, :]==0])
    mean_1 = compute_mean(data[:, labels[0, :]==1])
    variance = compute_variance(data)
    cov = compute_cov(data)
    skew = ss.skew(data)

    for i in range(len(feature_names)):
        print()
        print("Feature %d, %s has the following properties:" % (i, feature_names[i]))
        print()
        print("Mean is %3f" % (mean[i]))
        print("Mean computed for class %d, is %3f" % (0, mean_0[i]))
        print("Mean computed for class %d, is %3f" % (1, mean_1[i]))
        print()
        print("Variance is %3f" % (variance[i]))
        print()
        print("Skewness is %3f" % (skew[i]))
        print() 
        print("/////////////////////////////////////////////////////////////")
        print()

    print("The covariance matrix for the dataset is:")
    print(cov)
    print()
    print("/////////////////////////////////////////////////////////////")

    return mean, mean_0, mean_1, cov

# function to plot histograms of the features, to analyze feature distribution
def plot_hist(DTR, LTR, bool_save):

    feature_names = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", 
                "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]
    DTR_0 = DTR[:, LTR[0, :]==0]
    DTR_1 = DTR[:, LTR[0, :]==1]
    for i in range(DTR.shape[0]):
        plt.figure()
        plt.xlabel("Attribute %d - %s" % (i, feature_names[i]))
        plt.ylabel("Frequency")
        plt.hist(DTR_0[i, :], 15, density = True, label = 'Class 0', histtype='stepfilled', color = 'firebrick', alpha=0.7, linewidth=1)
        plt.hist(DTR_1[i, :], 15, density = True, label = 'Class 1', histtype='stepfilled', color = 'navy', alpha=0.7, linewidth=1)
        plt.legend()
        plt.tight_layout()
        plt.show()
        if bool_save:
            plt.savefig('./project/graphs/hist_%d.pdf' % i)

# function to plot boxplots of the features, to analyze feature distribution and highlight outliers
def plot_boxplot(DTR, LTR, bool_save):

    feature_names = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", 
                "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]
    DTR_0 = DTR[:, LTR[0, :]==0]
    DTR_1 = DTR[:, LTR[0, :]==1]
    for i in range(DTR.shape[0]):
        fig, ax = plt.subplots()
        #fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
        c0 = 'firebrick'
        c0b = 'indianred'
        c1 = 'navy'
        c1b = 'cornflowerblue'
        bp1 = ax.boxplot(DTR_0[i, :],
                    positions = [0],
                    notch=True, 
                    patch_artist=True,
                    boxprops=dict(facecolor=c0b, color=c0),
                    capprops=dict(color=c0),
                    whiskerprops=dict(color=c0),
                    flierprops=dict(color=c0, markeredgecolor=c0),
                    medianprops=dict(color=c0),
                    )

        bp2 = ax.boxplot(DTR_1[i, :],
                    positions = [1],
                    notch=True, 
                    patch_artist=True,
                    boxprops=dict(facecolor=c1b, color=c1),
                    capprops=dict(color=c1),
                    whiskerprops=dict(color=c1),
                    flierprops=dict(color=c1, markeredgecolor=c1),
                    medianprops=dict(color=c1),
                    )  
        #ax1.set_title('Class 0')
        #ax2.set_title('Class 1')
        #for ax in [ax1, ax2]:
        #ax.yaxis.grid(True)
        ax.set_xlabel('Attribute %d - %s' % (i, feature_names[i]))
        ax.set_ylabel('Observed values')
        red_patch = mpatches.Patch(color='firebrick', label='Class 0')
        blue_patch = mpatches.Patch(color='navy', label='Class 1')
        ax.legend(handles=[red_patch, blue_patch])
        # plt.tight_layout()
        plt.show()
        if bool_save:
            plt.savefig('./project/graphs/boxplot_%d.pdf' % i)

# function to plot 3d scatter of three features at the same time, potential use along with PCA = 3
def plot_3dscatter(data, labels, bool_save):

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(data[0, labels[0, :]==0], data[1, labels[0, :]==0], data[2, labels[0, :]==0], label = 'Class 0', alpha=0.2,)
    ax.scatter(data[0, labels[0, :]==1], data[1, labels[0, :]==1], data[2, labels[0, :]==1], label = 'Class 1', alpha=0.2,)

    ax.set_xlabel('First Principal Component')
    ax.set_ylabel('Second Principal Component')
    ax.set_zlabel('Third Principal Component')
    plt.legend()
    plt.tight_layout()
    plt.show()
    if bool_save:
            plt.savefig('./project/graphs/scatter3dpca.pdf')

# gaussianization process of the data
def gaussianization(data):

    # rank computation: gives each sample a rank compared to other samples
    rank_matrix = np.empty([data.shape[0], data.shape[1]])

    for i in range(data.shape[0]):
        rank_matrix[i, :] = ss.rankdata(data[i, :])
    
    rank_matrix = (rank_matrix)/(data.shape[1]+2)
    
    # ppf function from scipy.stats module / inverse of probability density function
    # hard to write, easier to solve numerically 
    gaussianized_data = norm.ppf(rank_matrix)

    return gaussianized_data

# function to perform Principal Component Analysis, with the desired number of components to extract as parameter
def pca(data, cov, num_components):
    
    # computation of the eigenvalues/eigenvectors: 
    # s contains the eigenvalues from smallest to largest
    # U contains the eigenvectors as columns
    s, U = np.linalg.eigh(cov)

    m = num_components
    # take the m principal components
    pc = U[:, -m:]
    
    # projecting the original datasets along these principal components
    DProjList = [] 

    for i in range(data.shape[1]):
        xi = mcol(data[:, i])
        yi = np.dot(pc.T, xi)
        DProjList.append(yi)

    # transformation into the new transform data matrix
    DY = np.hstack(DProjList)

    return DY

# compute logarithmic probability density function for each sample        
def logpdf_1sample(values, Mu, C):

    #computation of the logMultiVariateGaussian for each sample
    P = np.linalg.inv(C)
    logN = -0.5*values.shape[0]*np.log(2*np.pi)-0.5*np.linalg.slogdet(C)[1]-0.5*np.dot(np.dot((values-Mu).T, P), (values-Mu)) 

    return logN.ravel()

def logpdf_GAU_ND(X, mu, C):
    Y = [logpdf_1sample(X[:, i:i+1], mu, C) for i in range(X.shape[1])]
    return np.array(Y).ravel()

# transform the class conditional scores into log-likelihood ratios
def log_likelihood_ratio(log_scores):
    scores = np.exp(log_scores)
    llr = np.log(scores[1, :]/scores[0, :])
    return llr

# function to compute the optimal bayes decision based on threshold
def optimal_decision(llr, prior, Cfn, Cfp, threshold = None):
    
    if threshold is None:
        threshold = -np.log(prior * Cfn) + np.log((1-prior) * Cfp)

    label = llr > threshold

    return label

# compute values for the confusion matrix 
def confusion_matrix(pred, labels):
    conf = np.zeros([2,2])
    for i in range(2):  
        for j in range(2):
            conf[i, j] = ((pred == i ) * (labels == j)).sum()
    return conf

def compute_empirical_bayes(conf, prior, Cfn, Cfp):
    fnr = conf[0,1] / (conf[0,1] + conf[1,1])
    fpr = conf[1,0] / (conf[1,0] + conf[0,0])

    return prior * Cfn * fnr + (1-prior) * Cfp * fpr

def compute_normalized_bayes(conf, prior, Cfn, Cfp):
    emp = compute_empirical_bayes(conf, prior, Cfn, Cfp)
    return emp / min(prior * Cfn, (1-prior) * Cfp)

# compute values for the detection cost function, considering the normalized
def compute_DCF(scores, labels, prior, Cfn, Cfp, threshold = None):
    pred = optimal_decision(scores, prior, Cfn, Cfp, threshold = threshold)
    conf = confusion_matrix(pred, labels)
    return compute_normalized_bayes(conf, prior, Cfn, Cfp)

# compute the minimum DCF recalibrating the scores
def compute_min_DCF(scores, labels, prior, Cfn, Cfp):
    
    thresholds = np.array(scores)
    thresholds.sort()
    np.concatenate([np.array([-np.inf]), thresholds, np.array([np.inf])])
    dcf = []
    for th in thresholds:
        dcf.append(compute_DCF(scores, labels, prior, Cfn, Cfp, threshold = th))
    
    return np.array(dcf).min()

# roc diagram 
def roc(llr, labels, title, bool_save):
    thresholds = np.array(llr)
    thresholds.sort()
    thresholds = np.concatenate([np.array([-np.inf]), thresholds, np.array([np.inf])])
    FPR = np.zeros(thresholds.size)
    TPR = np.zeros(thresholds.size)

    for idx, t in enumerate(thresholds):
        pred = np.int32(llr > t)
        confusion = confusion_matrix(pred, labels)

        TPR[idx] = confusion[1,1] / (confusion[1,1] + confusion[0,1])
        FPR[idx] = confusion[1,0] / (confusion[1,0] + confusion[0,0])
    
    plt.figure()
    plt.xlabel("FPR - False positive rate")
    plt.ylabel("TPR - True Positive Rate")
    plt.title(title, fontsize = 10)
    plt.plot(FPR, TPR, color='navy')
    plt.grid(axis='x', color='0.95')
    plt.tight_layout()
    plt.show()
    if bool_save:
        plt.savefig('./project/graphs/roc_curve')

# bayes plots to verify effectiveness of classifiers 
def bayes_plots(scores, labels, parray, minCost = False):
    y = []
    for pi in parray:
        prior = 1 / (1 + np.exp(-pi))
        if minCost:
            y.append(compute_min_DCF(scores, labels, prior, 1, 1))
        else:
            y.append(compute_DCF(scores, labels, prior, 1, 1))
    
    return np.array(y)

def bayesplt(llr, LTE, title):
    parray = np.linspace(-3, 3, 21)
    bayes_dcf = bayes_plots(llr, LTE, parray, False)
    bayes_min = bayes_plots(llr, LTE, parray, True)
    plt.figure()
    plt.xlabel("p")
    plt.ylabel("DCF")
    plt.title(title)
    red_patch = mpatches.Patch(color='firebrick', label='DCF')
    blue_patch = mpatches.Patch(color='navy', label='min DCF')
    plt.legend(handles=[red_patch, blue_patch])
    plt.plot(parray, bayes_dcf, color='firebrick')
    plt.plot(parray, bayes_min, color='navy')
    plt.grid(axis='x', color='0.95')
    plt.tight_layout()
    plt.show()

# wrapper function for the various steps of MVG classifier
def multivariate_gaussian_classifier(DTR, LTR, DTE, LTE, datatype):

    DTR_0 = DTR[:, LTR[0, :]==0]
    DTR_1 = DTR[:, LTR[0, :]==1]
    lab = [0,1]
    mu_0 = np.empty([DTR.shape[0], 1])
    mu_0 = compute_mean(DTR_0)
    mu_1 = np.empty([DTR.shape[0], 1])
    mu_1 = compute_mean(DTR_1)

    cov_0 = compute_cov(DTR_0)
    cov_1 = compute_cov(DTR_1)
    
    log_scores = np.empty([len(lab), DTE.shape[1]])

    # compute log scores
    log_scores[0, :] = mrow(logpdf_GAU_ND(DTE, mu_0, cov_0))
    log_scores[1, :] = mrow(logpdf_GAU_ND(DTE, mu_1, cov_1))
    
    llr = log_likelihood_ratio(log_scores)

    # application parameters
    prior = 0.5
    Cfn = 1
    Cfp = 1
    min_DCF = compute_min_DCF(llr, LTE, prior, Cfn, Cfp)

    print("Application parameters: (Prior %1f, Cfn %d, Cfp %d)" % (prior, Cfn, Cfp))
    print("Minimum DCF: %f" % (min_DCF))
    print()

    prior = 0.8
    Cfn = 1
    Cfp = 1
    min_DCF = compute_min_DCF(llr, LTE, prior, Cfn, Cfp)

    print("Application parameters: (Prior %1f, Cfn %d, Cfp %d)" % (prior, Cfn, Cfp))
    print("Minimum DCF: %f" % (min_DCF))
    print()

    prior = 0.2
    Cfn = 1
    Cfp = 1
    min_DCF = compute_min_DCF(llr, LTE, prior, Cfn, Cfp)

    print("Application parameters: (Prior %1f, Cfn %d, Cfp %d)" % (prior, Cfn, Cfp))
    print("Minimum DCF: %f" % (min_DCF))
    print()

    prior = 0.5
    Cfn = 10
    Cfp = 1
    min_DCF = compute_min_DCF(llr, LTE, prior, Cfn, Cfp)

    print("Application parameters: (Prior %1f, Cfn %d, Cfp %d)" % (prior, Cfn, Cfp))
    print("Minimum DCF: %f" % (min_DCF))
    print()

    prior = 0.5
    Cfn = 1
    Cfp = 10
    min_DCF = compute_min_DCF(llr, LTE, prior, Cfn, Cfp)

    print("Application parameters: (Prior %1f, Cfn %d, Cfp %d)" % (prior, Cfn, Cfp))
    print("Minimum DCF: %f" % (min_DCF))
    print()

    if (see_graphs):
        roc(llr, LTE, "Multivariate Gaussian Classifier ROC graph" + " - " + datatype, False)
        bayesplt(llr, LTE, "Multivariate Gaussian Classifier bayes plot" + " - " + datatype)
        

# wrapper function for the various steps of MVG classifier with Naive Bayes hypothesis
def naive_multivariate_gaussian_classifier(DTR, LTR, DTE, LTE, datatype):

    DTR_0 = DTR[:, LTR[0, :]==0]
    DTR_1 = DTR[:, LTR[0, :]==1]
    lab = [0,1]
    mu_0 = np.empty([DTR.shape[0], 1])
    mu_0 = compute_mean(DTR_0)
    mu_1 = np.empty([DTR.shape[0], 1])
    mu_1 = compute_mean(DTR_1)

    cov_0 = compute_cov(DTR_0) * np.eye(DTR.shape[0])
    cov_1 = compute_cov(DTR_1) * np.eye(DTR.shape[0])

    
    log_scores = np.empty([len(lab), DTE.shape[1]])

    # compute log scores
    log_scores[0, :] = mrow(logpdf_GAU_ND(DTE, mu_0, cov_0))
    log_scores[1, :] = mrow(logpdf_GAU_ND(DTE, mu_1, cov_1))
    
    llr = log_likelihood_ratio(log_scores)

    # application parameters
    prior = 0.5
    Cfn = 1
    Cfp = 1
    min_DCF = compute_min_DCF(llr, LTE, prior, Cfn, Cfp)

    print("Application parameters: (Prior %1f, Cfn %d, Cfp %d)" % (prior, Cfn, Cfp))
    print("Minimum DCF: %f" % (min_DCF))
    print()

    prior = 0.8
    Cfn = 1
    Cfp = 1
    min_DCF = compute_min_DCF(llr, LTE, prior, Cfn, Cfp)

    print("Application parameters: (Prior %1f, Cfn %d, Cfp %d)" % (prior, Cfn, Cfp))
    print("Minimum DCF: %f" % (min_DCF))
    print()

    prior = 0.2
    Cfn = 1
    Cfp = 1
    min_DCF = compute_min_DCF(llr, LTE, prior, Cfn, Cfp)

    print("Application parameters: (Prior %1f, Cfn %d, Cfp %d)" % (prior, Cfn, Cfp))
    print("Minimum DCF: %f" % (min_DCF))
    print()

    prior = 0.5
    Cfn = 10
    Cfp = 1
    min_DCF = compute_min_DCF(llr, LTE, prior, Cfn, Cfp)

    print("Application parameters: (Prior %1f, Cfn %d, Cfp %d)" % (prior, Cfn, Cfp))
    print("Minimum DCF: %f" % (min_DCF))
    print()

    prior = 0.5
    Cfn = 1
    Cfp = 10
    min_DCF = compute_min_DCF(llr, LTE, prior, Cfn, Cfp)

    print("Application parameters: (Prior %1f, Cfn %d, Cfp %d)" % (prior, Cfn, Cfp))
    print("Minimum DCF: %f" % (min_DCF))
    print()


    if (see_graphs):
        roc(llr, LTE, "Naive Bayes Multivariate Gaussian Classifier ROC graph" + " - " + datatype, False)
        bayesplt(llr, LTE, "Naive Bayes Multivariate Gaussian Classifier bayes plot" + " - " + datatype)

# wrapper function for the various steps of MVG classifier with Tied covariances matrix
def tiedcov_multivariate_gaussian_classifier(DTR, LTR, DTE, LTE, datatype):

    DTR_0 = DTR[:, LTR[0, :]==0]
    DTR_1 = DTR[:, LTR[0, :]==1]
    lab = [0,1]
    mu_0 = np.empty([DTR.shape[0], 1])
    mu_0 = compute_mean(DTR_0)
    mu_1 = np.empty([DTR.shape[0], 1])
    mu_1 = compute_mean(DTR_1)

    cov = compute_cov(DTR)
    
    log_scores = np.empty([len(lab), DTE.shape[1]])

    # compute log scores
    log_scores[0, :] = mrow(logpdf_GAU_ND(DTE, mu_0, cov))
    log_scores[1, :] = mrow(logpdf_GAU_ND(DTE, mu_1, cov))
    
    llr = log_likelihood_ratio(log_scores)
    
    # application parameters
    prior = 0.5
    Cfn = 1
    Cfp = 1
    min_DCF = compute_min_DCF(llr, LTE, prior, Cfn, Cfp)

    print("Application parameters: (Prior %1f, Cfn %d, Cfp %d)" % (prior, Cfn, Cfp))
    print("Minimum DCF: %f" % (min_DCF))
    print()

    prior = 0.8
    Cfn = 1
    Cfp = 1
    min_DCF = compute_min_DCF(llr, LTE, prior, Cfn, Cfp)

    print("Application parameters: (Prior %1f, Cfn %d, Cfp %d)" % (prior, Cfn, Cfp))
    print("Minimum DCF: %f" % (min_DCF))
    print()

    prior = 0.2
    Cfn = 1
    Cfp = 1
    min_DCF = compute_min_DCF(llr, LTE, prior, Cfn, Cfp)

    print("Application parameters: (Prior %1f, Cfn %d, Cfp %d)" % (prior, Cfn, Cfp))
    print("Minimum DCF: %f" % (min_DCF))
    print()

    prior = 0.5
    Cfn = 10
    Cfp = 1
    min_DCF = compute_min_DCF(llr, LTE, prior, Cfn, Cfp)

    print("Application parameters: (Prior %1f, Cfn %d, Cfp %d)" % (prior, Cfn, Cfp))
    print("Minimum DCF: %f" % (min_DCF))
    print()

    prior = 0.5
    Cfn = 1
    Cfp = 10
    min_DCF = compute_min_DCF(llr, LTE, prior, Cfn, Cfp)

    print("Application parameters: (Prior %1f, Cfn %d, Cfp %d)" % (prior, Cfn, Cfp))
    print("Minimum DCF: %f" % (min_DCF))
    print()

    if (see_graphs):
        roc(llr, LTE, "Multivariate Gaussian Classifier with Tied Covariance ROC graph" + " - " + datatype, False)
        bayesplt(llr, LTE, "Multivariate Gaussian Classifier with Tied Covariance bayes plot" + " - " + datatype)

# wrapper function for the various steps of MVG classifier with Naive Bayes hypothesis and tied covariance matrix
def tiednaive_multivariate_gaussian_classifier(DTR, LTR, DTE, LTE, datatype):

    DTR_0 = DTR[:, LTR[0, :]==0]
    DTR_1 = DTR[:, LTR[0, :]==1]
    lab = [0,1]
    mu_0 = np.empty([DTR.shape[0], 1])
    mu_0 = compute_mean(DTR_0)
    mu_1 = np.empty([DTR.shape[0], 1])
    mu_1 = compute_mean(DTR_1)

    cov = compute_cov(DTR) * np.eye(DTR.shape[0])
    
    log_scores = np.empty([len(lab), DTE.shape[1]])

    # compute log scores
    log_scores[0, :] = mrow(logpdf_GAU_ND(DTE, mu_0, cov))
    log_scores[1, :] = mrow(logpdf_GAU_ND(DTE, mu_1, cov))
    
    llr = log_likelihood_ratio(log_scores)
    
    # application parameters
    prior = 0.5
    Cfn = 1
    Cfp = 1
    min_DCF = compute_min_DCF(llr, LTE, prior, Cfn, Cfp)

    print("Application parameters: (Prior %1f, Cfn %d, Cfp %d)" % (prior, Cfn, Cfp))
    print("Minimum DCF: %f" % (min_DCF))
    print()

    prior = 0.8
    Cfn = 1
    Cfp = 1
    min_DCF = compute_min_DCF(llr, LTE, prior, Cfn, Cfp)

    print("Application parameters: (Prior %1f, Cfn %d, Cfp %d)" % (prior, Cfn, Cfp))
    print("Minimum DCF: %f" % (min_DCF))
    print()

    prior = 0.2
    Cfn = 1
    Cfp = 1
    min_DCF = compute_min_DCF(llr, LTE, prior, Cfn, Cfp)

    print("Application parameters: (Prior %1f, Cfn %d, Cfp %d)" % (prior, Cfn, Cfp))
    print("Minimum DCF: %f" % (min_DCF))
    print()

    prior = 0.5
    Cfn = 10
    Cfp = 1
    min_DCF = compute_min_DCF(llr, LTE, prior, Cfn, Cfp)

    print("Application parameters: (Prior %1f, Cfn %d, Cfp %d)" % (prior, Cfn, Cfp))
    print("Minimum DCF: %f" % (min_DCF))
    print()

    prior = 0.5
    Cfn = 1
    Cfp = 10
    min_DCF = compute_min_DCF(llr, LTE, prior, Cfn, Cfp)

    print("Application parameters: (Prior %1f, Cfn %d, Cfp %d)" % (prior, Cfn, Cfp))
    print("Minimum DCF: %f" % (min_DCF))
    print()

    if (see_graphs):
        roc(llr, LTE, "Naive Bayes Multivariate Gaussian Classifier with Tied Covariance ROC graph" + " - " + datatype, False)
        bayesplt(llr, LTE, "Naive Bayes Multivariate Gaussian Classifier with Tied Covariance bayes plot" + " - " + datatype)

# wrapper function for logistic regression
def logreg_wrapper(DTR, LTR, l):
    labels = LTR * 2.0 - 1.0
    num_features = DTR.shape[0]
    def logreg(v):
        w = mcol(v[0:num_features])
        b = v[-1]
        S = np.dot(w.T, DTR) + b
        cxe = np.logaddexp(0, -S*labels).mean()
        return cxe + 0.5*l * np.linalg.norm(w)**2
    return logreg

# function to perform the classification with logistic regression, transforming the scores into scores with
# probability interpretation, to be able to perform Min DCF computation
def logistic_regression(DTR, LTR, DTE, LTE, lamb, datatype, crossval = False):
    logreg_obj = logreg_wrapper(DTR, LTR, lamb)
    _v, _J, _d = so.fmin_l_bfgs_b(logreg_obj, np.zeros(DTR.shape[0]+1), approx_grad = True)
    _w = _v[0:DTR.shape[0]]
    _b = _v[-1]
    scores_posteriors = np.dot(_w.T, DTE) + _b

    empirical_prior = (LTR[0, :]==1).sum(0)/LTR.shape[1]

    scores = scores_posteriors - log(empirical_prior/(1-empirical_prior))
    # application parameters
    prior = 0.5
    Cfn = 1
    Cfp = 1
    min_DCF = compute_min_DCF(scores, LTE, prior, Cfn, Cfp)

    print("Application parameters: (Prior %1f, Cfn %d, Cfp %d)" % (prior, Cfn, Cfp))
    print("Minimum DCF: %f" % (min_DCF))
    print()

    prior = 0.8
    Cfn = 1
    Cfp = 1
    min_DCF = compute_min_DCF(scores, LTE, prior, Cfn, Cfp)

    print("Application parameters: (Prior %1f, Cfn %d, Cfp %d)" % (prior, Cfn, Cfp))
    print("Minimum DCF: %f" % (min_DCF))
    print()

    prior = 0.2
    Cfn = 1
    Cfp = 1
    min_DCF = compute_min_DCF(scores, LTE, prior, Cfn, Cfp)

    print("Application parameters: (Prior %1f, Cfn %d, Cfp %d)" % (prior, Cfn, Cfp))
    print("Minimum DCF: %f" % (min_DCF))
    print()

    prior = 0.5
    Cfn = 10
    Cfp = 1
    min_DCF = compute_min_DCF(scores, LTE, prior, Cfn, Cfp)

    print("Application parameters: (Prior %1f, Cfn %d, Cfp %d)" % (prior, Cfn, Cfp))
    print("Minimum DCF: %f" % (min_DCF))
    print()

    prior = 0.5
    Cfn = 1
    Cfp = 10
    min_DCF = compute_min_DCF(scores, LTE, prior, Cfn, Cfp)

    print("Application parameters: (Prior %1f, Cfn %d, Cfp %d)" % (prior, Cfn, Cfp))
    print("Minimum DCF: %f" % (min_DCF))
    print()

    prior = empirical_prior
    Cfn = 1
    Cfp = 1
    min_DCF = compute_min_DCF(scores, LTE, prior, Cfn, Cfp)

    
    print("Empirical prior test:")
    print("Application parameters: (Prior %1f, Cfn %d, Cfp %d)" % (prior, Cfn, Cfp))
    print("Minimum DCF: %f" % (min_DCF))
    print()


    if(see_graphs):
        roc(scores, LTE, "Logistic Regression ROC graph" + " - " + datatype, False)
        bayesplt(scores, LTE, "Logistic Regression bayes plot" + " - " + datatype)

    if crossval:
        return min_DCF

# Support Vector Machine linear classifier trainer
def svm_linear(DTR, LTR, C, K = 1):

    DTREXT = np.vstack([DTR, np.ones(DTR.shape[1])*K])

    z = np.zeros(LTR.shape)
    z[LTR == 1] = 1
    z[LTR == 0] = -1

    # Hij matrix, second operation exploits broadcasting to perform the multiplication
    H = np.dot(DTREXT.T, DTREXT)
    H = mcol(z)*mrow(z)*H

    # alpha is the Lagrangian multiplier
    def Jdual(alpha):
        Ha = np.dot(H, mcol(alpha))
        aHa = np.dot(mrow(alpha), Ha)
        a1 = alpha.sum()

        return -0.5 * aHa.ravel() + a1,  -Ha.ravel() + np.ones(alpha.size)

    def Ldual(alpha):
        loss, grad = Jdual(alpha)
        return -loss, -grad

    def Jprimal(w):
        S = np.dot(mrow(w), DTREXT)
        loss = np.maximum(np.zeros(S.shape), 1-z*S).sum()
        return 0.5 * np.linalg.norm(w)**2 + C * loss
        
    alphastar, _x, _y = so.fmin_l_bfgs_b(
        Ldual,
        np.zeros(DTR.shape[1]),
        bounds = [(0, C)] * DTR.shape[1],
        factr = 0.0,
        maxiter = 100000,
        maxfun = 100000,
        )

    wstar = np.dot(DTREXT, mcol(alphastar)* mcol(z))

    return wstar

def svm_RBF(DTR, LTR, C, gamma, K = 1):

    z = np.zeros(LTR.shape)
    z[LTR == 1] = 1
    z[LTR == 0] = -1

    # Hij matrix, second operation exploits broadcasting to perform the multiplication
    Dist = mcol((DTR**2).sum(0)) + mrow((DTR**2).sum(0))-2*np.dot(DTR.T, DTR)
    H = np.exp(-gamma*Dist) + K**2
    H = mcol(z)*mrow(z)*H

    # alpha is the Lagrangian multiplier
    def Jdual(alpha):
        Ha = np.dot(H, mcol(alpha))
        aHa = np.dot(mrow(alpha), Ha)
        a1 = alpha.sum()

        return -0.5 * aHa.ravel() + a1,  -Ha.ravel() + np.ones(alpha.size)

    def Ldual(alpha):
        loss, grad = Jdual(alpha)
        return -loss, -grad
        
    alphastar, _x, _y = so.fmin_l_bfgs_b(
        Ldual,
        np.zeros(DTR.shape[1]),
        bounds = [(0, C)] * DTR.shape[1],
        factr = 0,
        maxiter = 400000,
        maxfun = 200000,
        )

    return alphastar, z

def svm_poly(DTR, LTR, C, c, d, K = 1):

    z = np.zeros(LTR.shape)
    z[LTR == 1] = 1
    z[LTR == 0] = -1

    # Hij matrix, second operation exploits broadcasting to perform the multiplication
    xTx = np.dot(DTR.T, DTR) + c
    H = xTx**d + K**2
    H = mcol(z)*mrow(z)*H

    # alpha is the Lagrangian multiplier
    def Jdual(alpha):
        Ha = np.dot(H, mcol(alpha))
        aHa = np.dot(mrow(alpha), Ha)
        a1 = alpha.sum()

        return -0.5 * aHa.ravel() + a1,  -Ha.ravel() + np.ones(alpha.size)

    def Ldual(alpha):
        loss, grad = Jdual(alpha)
        return -loss, -grad
        
    alphastar, _x, _y = so.fmin_l_bfgs_b(
        Ldual,
        np.zeros(DTR.shape[1]),
        bounds = [(0, C)] * DTR.shape[1],
        factr = 0,
        maxiter = 400000,
        maxfun = 200000,
        )

    return alphastar, z

# Support Vector Machine scorer
def svm_scoring(DTE, LTE, wstar, K):

    wstar = np.array(wstar)
    wstar = mcol(wstar)
    DTEEXT = np.vstack([DTE, np.ones(DTE.shape[1])*K])
    scores = np.dot(wstar.T, DTEEXT)
    scores_array = scores[0, :]
    # pseudo-application parameters
    prior = 0.5
    Cfn = 1
    Cfp = 1
    min_DCF = compute_min_DCF(scores_array, LTE, prior, Cfn, Cfp)

    print("SVM Scoring")
    print("Minimum DCF: %f" % (min_DCF))
    print()

# Support Vector Machine scorer
def svm_RBF_scoring(DTE, LTE, DTR, K, gamma, alphastar, z):

    scores = np.zeros(DTE.shape[1])
    Dist = np.zeros(shape = (DTE.shape[1], DTR.shape[1]))

    for i in range(DTE.shape[1]):
        scoresToSum = 0
        for j in range(DTR.shape[1]):
            xi = DTE[:, i]
            xj = DTR[:, j]
            Dist[i, j] = np.linalg.norm(xi-xj)**2
            toSum = alphastar[j]*z[0, j]*(np.exp(-gamma*Dist[i,j])+K**2)
            scoresToSum = scoresToSum + toSum
        scores[i] = scoresToSum
    
    # pseudo-application parameters
    prior = 0.5
    Cfn = 1
    Cfp = 1
    min_DCF = compute_min_DCF(scores, LTE, prior, Cfn, Cfp)

    print("SVM Scoring")
    print("Minimum DCF: %f" % (min_DCF))
    print()

# Support Vector Machine scorer
def svm_poly_scoring(DTE, LTE, DTR, K, c, d, alphastar, z):

    scores = np.zeros(DTE.shape[1])
    Dist = np.zeros(shape = (DTE.shape[1], DTR.shape[1]))

    for i in range(DTE.shape[1]):
        scoresToSum = 0
        for j in range(DTR.shape[1]):
            xi = DTE[:, i]
            xj = DTR[:, j]
            Dist[i, j] = ((np.dot(xj.T, xi) + c))**d + K**2
            toSum = alphastar[j]*z[0, j]*Dist[i, j]
            scoresToSum = scoresToSum + toSum
        scores[i] = scoresToSum
    
    # pseudo-application parameters
    prior = 0.5
    Cfn = 1
    Cfp = 1
    min_DCF = compute_min_DCF(scores, LTE, prior, Cfn, Cfp)

    print("SVM Scoring")
    print("Minimum DCF: %f" % (min_DCF))
    print()


# Gaussian Mixture Model per sample log-likelihood computation 
def GMM_ll_perSample(X, gmm):

    G = len(gmm)
    N = X.shape[1]
    S = np.zeros((G, N))
    for g in range(G):
        S[g, :] = logpdf_GAU_ND(X, gmm[g][1], gmm[g][2]) + np.log(gmm[g][0])
    return sp.logsumexp(S, axis = 0)

# Gaussian Mixture Model EM algorithm
def GMM_EM(X, gmm):
    llNew = None
    llOld = None
    G = len(gmm)
    N = X.shape[1]

    while llOld is None or llNew - llOld > 1e-6:
        llOld = llNew
        SJ = np.zeros((G, N))
        for g in range(G):
            SJ[g, :] = logpdf_GAU_ND(X, gmm[g][1], gmm[g][2]) + np.log(gmm[g][0])
        SM = sp.logsumexp(SJ, axis = 0)
        llNew = SM.sum()/N
        P = np.exp(SJ - SM)
        gmmNew = []

        for g in range(G):
            gamma = P[g, :]
            Z = gamma.sum()
            F = (mrow(gamma)*X).sum(1)
            S = np.dot(X, (mrow(gamma)*X).T)
            w = Z/N
            mu = mcol(F/N)
            Sigma = S/N - np.dot(mu, mu.T)
            Sigma = eigen_constraint(Sigma)

            gmmNew.append([w, mu, Sigma])

        gmm = gmmNew
    
    return gmm 

# Gaussian Mixture Model EM algorithm with diagonal covariance matrix hypothesis
def GMM_EM_diagonal(X, gmm):
    llNew = None
    llOld = None
    G = len(gmm)
    N = X.shape[1]

    while llOld is None or llNew - llOld > 1e-6:
        llOld = llNew
        SJ = np.zeros((G, N))
        for g in range(G):
            SJ[g, :] = logpdf_GAU_ND(X, gmm[g][1], gmm[g][2]) + np.log(gmm[g][0])
        SM = sp.logsumexp(SJ, axis = 0)
        llNew = SM.sum()/N
        P = np.exp(SJ - SM)
        gmmNew = []

        for g in range(G):
            gamma = P[g, :]
            Z = gamma.sum()
            F = (mrow(gamma)*X).sum(1)
            S = np.dot(X, (mrow(gamma)*X).T)
            w = Z/N
            mu = mcol(F/N)
            Sigma = S/N - np.dot(mu, mu.T)

            Sigma = Sigma * np.eye(Sigma.shape[0])
            Sigma = eigen_constraint(Sigma)

            gmmNew.append([w, mu, Sigma])

        gmm = gmmNew
    
    return gmm 

# Gaussian Mixture Model EM algorithm with tied covariance matrix hypothesis
def GMM_EM_tied(X, gmm):
    llNew = None
    llOld = None
    G = len(gmm)
    N = X.shape[1]

    while llOld is None or llNew - llOld > 1e-6:
        llOld = llNew
        SJ = np.zeros((G, N))
        for g in range(G):
            SJ[g, :] = logpdf_GAU_ND(X, gmm[g][1], gmm[g][2]) + np.log(gmm[g][0])
        SM = sp.logsumexp(SJ, axis = 0)
        llNew = SM.sum()/N
        P = np.exp(SJ - SM)
        gmmNew = []

        for g in range(G):
            gamma = P[g, :]
            Z = gamma.sum()
            F = (mrow(gamma)*X).sum(1)
            S = np.dot(X, (mrow(gamma)*X).T)
            w = Z/N
            mu = mcol(F/N)
            Sigma = S/N - np.dot(mu, mu.T)
           

            gmmNew.append([w, mu, Sigma])
        
        Sigma_tied = np.empty_like(gmmNew[0][2])

        for g in range(G):
            Sigma_tied = Z*gmmNew[g][2] + Sigma_tied

        Sigma_tied = Sigma_tied/N
        Sigma_tied = eigen_constraint(Sigma_tied)

        for g in range(G):
            gmmNew[g][2] = Sigma_tied
            
        gmm = gmmNew
    
    return gmm 

# Gaussian Mixture Model LBG algorithm to generate initial gaussian components
def GMM_lbg(gmm, iter):

    gmm_new = []
    alpha = 0.1

    if iter:
        gmm_1 = np.array(np.empty(3), dtype=object)
        gmm_2 = np.array(np.empty(3), dtype=object)

        gmm_1[0]= gmm[0]/2
        gmm_2[0]= gmm[0]/2
    
        U, s, Vh = np.linalg.svd(gmm[2])
        d = U[:, 0:1] * s[0]**0.5 * alpha

        gmm_1[1]= gmm[1] + d
        gmm_2[1]= gmm[1] - d

        gmm_1[2] = gmm[2]
        gmm_2[2] = gmm[2]

        gmm_new.append(gmm_1)
        gmm_new.append(gmm_2)

    else:

        for g in range(len(gmm)):

            gmm_1 = np.array(np.empty(3), dtype=object)
            gmm_2 = np.array(np.empty(3), dtype=object)

            gmm_1[0]= gmm[g][0]/2
            gmm_2[0]= gmm[g][0]/2
        
            U, s, Vh = np.linalg.svd(gmm[g][2])
            d = U[:, 0:1] * s[0]**0.5 * alpha

            gmm_1[1]= gmm[g][1] + d
            gmm_2[1]= gmm[g][1] - d

            gmm_1[2] = gmm[g][2]
            gmm_2[2] = gmm[g][2]

            gmm_new.append(gmm_1)
            gmm_new.append(gmm_2)
    
    return gmm_new

# eigenvalues constraints to avoid degenerate solutions
def eigen_constraint(cov):

    psi = 0.01

    U, s, _ = np.linalg.svd(cov)
    s[s<psi] = psi
    cov = np.dot(U, mcol(s)*U.T)

    return cov


# GMM classifier wrapper function
def GMM_classifier(DTR, LTR, DTE, LTE, datatype ):

    DTR_0 = DTR[:, LTR[0, :]==0]
    DTR_1 = DTR[:, LTR[0, :]==1]

    lab = [0,1]

    mu_0 = np.empty([DTR.shape[0], 1])
    mu_0 = compute_mean(DTR_0)
    mu_1 = np.empty([DTR.shape[0], 1])
    mu_1 = compute_mean(DTR_1)

    cov_0 = eigen_constraint(compute_cov(DTR_0))
    cov_1 = eigen_constraint(compute_cov(DTR_1))
    
    gmm_class0 = np.array([1, mu_0, cov_0], dtype=object)
    gmm_class0 = GMM_lbg(gmm_class0, True)
    gmm_class0 = GMM_EM(DTR_0, gmm_class0)
    gmm_class0 = GMM_lbg(gmm_class0, False)
    gmm_class0 = GMM_EM(DTR_0, gmm_class0)
    gmm_class0 = GMM_lbg(gmm_class0, False)
    gmm_class0 = GMM_EM(DTR_0, gmm_class0)

    gmm_class1 = np.array([1, mu_1, cov_1], dtype=object)
    gmm_class1 = GMM_lbg(gmm_class1, True)
    gmm_class1 = GMM_EM(DTR_1, gmm_class1)
    gmm_class1 = GMM_lbg(gmm_class1, False)
    gmm_class1 = GMM_EM(DTR_1, gmm_class1)
    gmm_class1 = GMM_lbg(gmm_class1, False)
    gmm_class1 = GMM_EM(DTR_1, gmm_class1)

    log_scores = np.empty([len(lab), DTE.shape[1]])

    log_scores[0, :] = mrow(GMM_ll_perSample(DTE, gmm_class0))
    log_scores[1, :] = mrow(GMM_ll_perSample(DTE, gmm_class1))

    llr = log_likelihood_ratio(log_scores)
    
    # application parameters
    prior = 0.5
    Cfn = 1
    Cfp = 1
    min_DCF = compute_min_DCF(llr, LTE, prior, Cfn, Cfp)

    print("Application parameters: (Prior %1f, Cfn %d, Cfp %d)" % (prior, Cfn, Cfp))
    print("Minimum DCF: %f" % (min_DCF))
    print()

    prior = 0.8
    Cfn = 1
    Cfp = 1
    min_DCF = compute_min_DCF(llr, LTE, prior, Cfn, Cfp)

    print("Application parameters: (Prior %1f, Cfn %d, Cfp %d)" % (prior, Cfn, Cfp))
    print("Minimum DCF: %f" % (min_DCF))
    print()

    prior = 0.2
    Cfn = 1
    Cfp = 1
    min_DCF = compute_min_DCF(llr, LTE, prior, Cfn, Cfp)

    print("Application parameters: (Prior %1f, Cfn %d, Cfp %d)" % (prior, Cfn, Cfp))
    print("Minimum DCF: %f" % (min_DCF))
    print()

    prior = 0.5
    Cfn = 10
    Cfp = 1
    min_DCF = compute_min_DCF(llr, LTE, prior, Cfn, Cfp)

    print("Application parameters: (Prior %1f, Cfn %d, Cfp %d)" % (prior, Cfn, Cfp))
    print("Minimum DCF: %f" % (min_DCF))
    print()

    prior = 0.5
    Cfn = 1
    Cfp = 10
    min_DCF = compute_min_DCF(llr, LTE, prior, Cfn, Cfp)

    print("Application parameters: (Prior %1f, Cfn %d, Cfp %d)" % (prior, Cfn, Cfp))
    print("Minimum DCF: %f" % (min_DCF))
    print()

# GMM classifier with naive Bayes hypothesis wrapper function
def GMM_naivebayes_classifier(DTR, LTR, DTE, LTE, datatype):

    DTR_0 = DTR[:, LTR[0, :]==0]
    DTR_1 = DTR[:, LTR[0, :]==1]

    lab = [0,1]

    mu_0 = np.empty([DTR.shape[0], 1])
    mu_0 = compute_mean(DTR_0)
    mu_1 = np.empty([DTR.shape[0], 1])
    mu_1 = compute_mean(DTR_1)

    cov_0 = eigen_constraint(compute_cov(DTR_0))
    cov_1 = eigen_constraint(compute_cov(DTR_1))
    
    gmm_class0 = np.array([1, mu_0, cov_0], dtype=object)
    gmm_class0 = GMM_lbg(gmm_class0, True)
    gmm_class0 = GMM_EM_diagonal(DTR_0, gmm_class0)
    gmm_class0 = GMM_lbg(gmm_class0, False)
    gmm_class0 = GMM_EM_diagonal(DTR_0, gmm_class0)
    gmm_class0 = GMM_lbg(gmm_class0, False)
    gmm_class0 = GMM_EM_diagonal(DTR_0, gmm_class0)

    gmm_class1 = np.array([1, mu_1, cov_1], dtype=object)
    gmm_class1 = GMM_lbg(gmm_class1, True)
    gmm_class1 = GMM_EM_diagonal(DTR_1, gmm_class1)
    gmm_class1 = GMM_lbg(gmm_class1, False)
    gmm_class1 = GMM_EM_diagonal(DTR_1, gmm_class1)
    gmm_class1 = GMM_lbg(gmm_class1, False)
    gmm_class1 = GMM_EM_diagonal(DTR_1, gmm_class1)

    log_scores = np.empty([len(lab), DTE.shape[1]])

    log_scores[0, :] = mrow(GMM_ll_perSample(DTE, gmm_class0))
    log_scores[1, :] = mrow(GMM_ll_perSample(DTE, gmm_class1))

    llr = log_likelihood_ratio(log_scores)
    
    # application parameters
    prior = 0.5
    Cfn = 1
    Cfp = 1
    min_DCF = compute_min_DCF(llr, LTE, prior, Cfn, Cfp)

    print("Application parameters: (Prior %1f, Cfn %d, Cfp %d)" % (prior, Cfn, Cfp))
    print("Minimum DCF: %f" % (min_DCF))
    print()

    prior = 0.8
    Cfn = 1
    Cfp = 1
    min_DCF = compute_min_DCF(llr, LTE, prior, Cfn, Cfp)

    print("Application parameters: (Prior %1f, Cfn %d, Cfp %d)" % (prior, Cfn, Cfp))
    print("Minimum DCF: %f" % (min_DCF))
    print()

    prior = 0.2
    Cfn = 1
    Cfp = 1
    min_DCF = compute_min_DCF(llr, LTE, prior, Cfn, Cfp)

    print("Application parameters: (Prior %1f, Cfn %d, Cfp %d)" % (prior, Cfn, Cfp))
    print("Minimum DCF: %f" % (min_DCF))
    print()

    prior = 0.5
    Cfn = 10
    Cfp = 1
    min_DCF = compute_min_DCF(llr, LTE, prior, Cfn, Cfp)

    print("Application parameters: (Prior %1f, Cfn %d, Cfp %d)" % (prior, Cfn, Cfp))
    print("Minimum DCF: %f" % (min_DCF))
    print()

    prior = 0.5
    Cfn = 1
    Cfp = 10
    min_DCF = compute_min_DCF(llr, LTE, prior, Cfn, Cfp)

    print("Application parameters: (Prior %1f, Cfn %d, Cfp %d)" % (prior, Cfn, Cfp))
    print("Minimum DCF: %f" % (min_DCF))
    print()

# GMM classifier with Tied Covariance wrapper function
def GMM_tiedcov_classifier(DTR, LTR, DTE, LTE, datatype):

    DTR_0 = DTR[:, LTR[0, :]==0]
    DTR_1 = DTR[:, LTR[0, :]==1]

    lab = [0,1]

    mu_0 = np.empty([DTR.shape[0], 1])
    mu_0 = compute_mean(DTR_0)
    mu_1 = np.empty([DTR.shape[0], 1])
    mu_1 = compute_mean(DTR_1)

    cov_0 = eigen_constraint(compute_cov(DTR_0))
    cov_1 = eigen_constraint(compute_cov(DTR_1))
    
    gmm_class0 = np.array([1, mu_0, cov_0], dtype=object)
    gmm_class0 = GMM_lbg(gmm_class0, True)
    gmm_class0 = GMM_EM_tied(DTR_0, gmm_class0)
    gmm_class0 = GMM_lbg(gmm_class0, False)
    gmm_class0 = GMM_EM_tied(DTR_0, gmm_class0)
    gmm_class0 = GMM_lbg(gmm_class0, False)
    gmm_class0 = GMM_EM_tied(DTR_0, gmm_class0)

    gmm_class1 = np.array([1, mu_1, cov_1], dtype=object)
    gmm_class1 = GMM_lbg(gmm_class1, True)
    gmm_class1 = GMM_EM_tied(DTR_1, gmm_class1)
    gmm_class1 = GMM_lbg(gmm_class1, False)
    gmm_class1 = GMM_EM_tied(DTR_1, gmm_class1)
    gmm_class1 = GMM_lbg(gmm_class1, False)
    gmm_class1 = GMM_EM_tied(DTR_1, gmm_class1)

    log_scores = np.empty([len(lab), DTE.shape[1]])

    log_scores[0, :] = mrow(GMM_ll_perSample(DTE, gmm_class0))
    log_scores[1, :] = mrow(GMM_ll_perSample(DTE, gmm_class1))

    llr = log_likelihood_ratio(log_scores)
    
    # application parameters
    prior = 0.5
    Cfn = 1
    Cfp = 1
    min_DCF = compute_min_DCF(llr, LTE, prior, Cfn, Cfp)

    print("Application parameters: (Prior %1f, Cfn %d, Cfp %d)" % (prior, Cfn, Cfp))
    print("Minimum DCF: %f" % (min_DCF))
    print()

    prior = 0.8
    Cfn = 1
    Cfp = 1
    min_DCF = compute_min_DCF(llr, LTE, prior, Cfn, Cfp)

    print("Application parameters: (Prior %1f, Cfn %d, Cfp %d)" % (prior, Cfn, Cfp))
    print("Minimum DCF: %f" % (min_DCF))
    print()

    prior = 0.2
    Cfn = 1
    Cfp = 1
    min_DCF = compute_min_DCF(llr, LTE, prior, Cfn, Cfp)

    print("Application parameters: (Prior %1f, Cfn %d, Cfp %d)" % (prior, Cfn, Cfp))
    print("Minimum DCF: %f" % (min_DCF))
    print()

    prior = 0.5
    Cfn = 10
    Cfp = 1
    min_DCF = compute_min_DCF(llr, LTE, prior, Cfn, Cfp)

    print("Application parameters: (Prior %1f, Cfn %d, Cfp %d)" % (prior, Cfn, Cfp))
    print("Minimum DCF: %f" % (min_DCF))
    print()

    prior = 0.5
    Cfn = 1
    Cfp = 10
    min_DCF = compute_min_DCF(llr, LTE, prior, Cfn, Cfp)

    print("Application parameters: (Prior %1f, Cfn %d, Cfp %d)" % (prior, Cfn, Cfp))
    print("Minimum DCF: %f" % (min_DCF))
    print()


if __name__ == '__main__':

    print()
    print("Analysis results for the wine classification problem.")
    print("Do you want to see graphs from the analysis?")
    txt = input("(y/n): ")
    if txt == "y":
        see_graphs = True
    else:
        see_graphs = False

    ''' 
    data sources
    '''
    train_fname = "./project/data/Train.txt"
    test_fname = "./project/data/Test.txt"
    
    '''
    data and labels loading
    '''
    DTR, LTR = load_set(train_fname)
    DTE, LTE = load_set(test_fname)
    feature_names = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", 
                    "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]

    '''
    data gaussianization
    '''
    DTR_g = gaussianization(DTR)
    DTE_g = gaussianization(DTE)

    '''
    heat map of correlation matrix
    '''
    if (see_graphs):
        correlation_matrix(DTR_g, feature_names)

    '''
    data histogram plotting, raw and gaussianized
    '''
    if (see_graphs):
        plot_hist(DTR, LTR, False)
        # plot_hist(DTR, LTR, True)
        plot_hist(DTR_g, LTR, False)
        # plot_hist(DTR_g, LTR, True)

    '''
    raw data boxplot
    '''
    if (see_graphs):
        plot_boxplot(DTR, LTR, False)
        # plot_boxplot(DTR, LTR, True)

    '''
    exploratory_data_analysis(DTR, LTR)
    '''
    mu, mu_0, mu_1, cov = eda(DTR, LTR, feature_names)

    '''
    principal component analysis 
    '''
    pca_DTR = pca(DTR, cov, 10)
    pca_DTR_g = gaussianization(pca_DTR)
    pca_DTE = pca(DTE, cov, 10)
    pca_DTE_g = gaussianization(pca_DTE)
    pca_DTR5 = pca(DTR, cov, 5)
    pca_DTE5 = pca(DTE, cov, 5) 
    
    '''
    multivariate gaussian classifier results
    '''
    print()
    print("MVG with raw data:")
    multivariate_gaussian_classifier(DTR, LTR, DTE, LTE, "raw data")
    print()
    print("MVG with gaussianized data:")
    multivariate_gaussian_classifier(DTR_g, LTR, DTE_g, LTE, "gaussianized data")
    print()
    print("MVG with PCA with components = 10 and raw data:")
    multivariate_gaussian_classifier(pca_DTR, LTR, pca_DTE, LTE, "PCA")
    print()
    print("MVG with PCA with components = 5 and raw data:")
    multivariate_gaussian_classifier(pca_DTR5, LTR, pca_DTE5, LTE, "PCA - m=5")
    print()
    print("MVG with PCA with components = 10 and gaussianized data:")
    multivariate_gaussian_classifier(pca_DTR_g, LTR, pca_DTE_g, LTE, "PCA and gaussianized data")
    print()
    print("/////////////////////////////////////////////////////////////")

    '''
    naive bayes multivariate gaussian classifier results
    '''
    print()
    print("Naive Bayes MVG with raw data:")
    naive_multivariate_gaussian_classifier(DTR, LTR, DTE, LTE, "raw data")
    print()
    print("Naive Bayes MVG with gaussianized data:")
    naive_multivariate_gaussian_classifier(DTR_g, LTR, DTE_g, LTE, "gaussianized data")
    print()
    print("Naive Bayes MVG with PCA with components = 10 and raw data:")
    naive_multivariate_gaussian_classifier(pca_DTR, LTR, pca_DTE, LTE, "PCA data")
    print()
    print("Naive Bayes MVG with PCA with components = 5 and raw data:")
    naive_multivariate_gaussian_classifier(pca_DTR5, LTR, pca_DTE5, LTE, "PCA data - m=5")
    print()
    print("Naive Bayes MVG with PCA with components = 10 and gaussianized data:")
    naive_multivariate_gaussian_classifier(pca_DTR_g, LTR, pca_DTE_g, LTE, "PCA and gaussianized data")
    print()
    print("/////////////////////////////////////////////////////////////")

    '''
    tied covariance multivariate gaussian classifier results
    '''
    print()
    print("Tied covariance MVG with raw data:")
    tiedcov_multivariate_gaussian_classifier(DTR, LTR, DTE, LTE, "raw data")
    print()
    print("Tied Covariance MVG with gaussianized data:")
    tiedcov_multivariate_gaussian_classifier(DTR_g, LTR, DTE_g, LTE, "gaussianized data")
    print()
    print("Tied Covariance MVG with PCA with components = 10 and raw data:")
    tiedcov_multivariate_gaussian_classifier(pca_DTR, LTR, pca_DTE, LTE, "PCA data")
    print()
    print()
    print("Tied Covariance MVG with PCA with components = 5 and raw data:")
    tiedcov_multivariate_gaussian_classifier(pca_DTR5, LTR, pca_DTE5, LTE, "PCA data - m=5")
    print()
    print("Tied Covariance MVG with PCA with components = 10 and gaussianized data:")
    tiedcov_multivariate_gaussian_classifier(pca_DTR_g, LTR, pca_DTE_g, LTE, "PCA and gaussianized data")
    print()
    print("/////////////////////////////////////////////////////////////")

    '''
    tied naive covariance multivariate gaussian classifier results
    '''
    print()
    print("Tied Naive covariance MVG with raw data:")
    tiednaive_multivariate_gaussian_classifier(DTR, LTR, DTE, LTE, "raw data")
    print()
    print("Tied Naive Covariance MVG with gaussianized data:")
    tiednaive_multivariate_gaussian_classifier(DTR_g, LTR, DTE_g, LTE, "gaussianized data")
    print()
    print("Tied Naive Covariance MVG with PCA with components = 10 and raw data:")
    tiednaive_multivariate_gaussian_classifier(pca_DTR, LTR, pca_DTE, LTE, "PCA data")
    print()
    print("Tied Naive Covariance MVG with PCA with components = 5 and raw data:")
    tiednaive_multivariate_gaussian_classifier(pca_DTR5, LTR, pca_DTE5, LTE, "PCA data - m=5")
    print()
    print("Tied Naive Covariance MVG with PCA with components = 10 and gaussianized data:")
    tiednaive_multivariate_gaussian_classifier(pca_DTR_g, LTR, pca_DTE_g, LTE, "PCA and gaussianized data")
    print()
    print("/////////////////////////////////////////////////////////////")

    print()
    print("Logistic Regression:")
    
    kf = KFold(n_splits=5, shuffle = True)

    lambda_toTest = [1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5, 1.0]
    minDCF = 1
    chosen_lamb = 10

    for lamb in lambda_toTest:
        
        print("Test on lambda = %2f" % lamb)
        dcf = 0

        for train_index, test_index in kf.split(DTR[0, :]):
            X_train, X_test = DTR[:, train_index], DTR[:, test_index]
            y_train, y_test = LTR[:, train_index], LTR[:, test_index]
            
            dcf = logistic_regression(X_train, y_train, X_test, y_test, lamb, "raw data", True) + dcf
        
        dcf = dcf / 5

        if dcf < minDCF:

            minDCF = dcf
            chosen_lamb = lamb
    
    print("Best performing lambda is %f" % chosen_lamb)
    
    '''
    print("Raw Data")
    for lamb in [1e-6, 1e-3, 0.1, 1.0]:
        print()
        print("")
        print("Lambda value: %2f" % lamb)
        logistic_regression(DTR, LTR, DTE, LTE, lamb, "raw data")
    
    print()
    print("/////////////////////////////////////////////////////////////")

    print()
    print("Gaussianized Data")
    for lamb in [1e-6, 1e-3, 0.1, 1.0]:
        print()
        print("")
        print("Lambda value: %2f" % lamb)
        logistic_regression(DTR_g, LTR, DTE_g, LTE, lamb, "gaussianized data")
    
    print()
    print("/////////////////////////////////////////////////////////////")

    print()
    print("Pca Data")
    for lamb in [1e-6, 1e-3, 0.1, 1.0]:
        print()
        print("Lambda value: %2f" % lamb)
        print()
        logistic_regression(pca_DTR, LTR, pca_DTE, LTE, lamb, "PCA data")
    
    print()
    print("/////////////////////////////////////////////////////////////")

    print()
    print("Pca Data m=5")
    for lamb in [1e-6, 1e-3, 0.1, 1.0]:
        print()
        print("Lambda value: %2f" % lamb)
        print()
        logistic_regression(pca_DTR5, LTR, pca_DTE5, LTE, lamb, "PCA data - m=5")
    '''
    print()
    print("/////////////////////////////////////////////////////////////")
    
    '''
    support vector machine classifier results
    '''
    '''
    print()
    print("SVM with raw data:")
    print()
    K = 5
    C = 0.1
    print("K = %f" % K) 
    print("C = %f" % C) 
    wstar = svm_linear(DTR, LTR, C, K)
    svm_scoring(DTE, LTE, wstar, K)
    print() 
    
    K = 1
    C = 0.2
    print("K = %f" % K) 
    print("C = %f" % C) 
    wstar = svm_linear(DTR, LTR, C, K)
    svm_scoring(DTE, LTE, wstar, K)
    print() 

    K = 1
    C = 0.5
    print("K = %f" % K) 
    print("C = %f" % C) 
    wstar = svm_linear(DTR, LTR, C, K)
    svm_scoring(DTE, LTE, wstar, K)
    print() 

    K = 1
    C = 1
    print("K = %d" % K) 
    print("C = %d" % C) 
    wstar = svm_linear(DTR, LTR, C, K)
    svm_scoring(DTE, LTE, wstar, K)
    print()
    
    print()
    print("SVM with PCA data:")
    print()
    K = 4
    C = 0.05
    print("K = %f" % K) 
    print("C = %f" % C) 
    wstar = svm_linear(pca_DTR, LTR, C, K)
    svm_scoring(pca_DTE, LTE, wstar, K)
    print() 

    K = 1
    C = 0.2
    print("K = %f" % K) 
    print("C = %f" % C) 
    wstar = svm_linear(pca_DTR, LTR, C, K)
    svm_scoring(pca_DTE, LTE, wstar, K)
    print() 

    K = 1
    C = 0.5
    print("K = %f" % K) 
    print("C = %f" % C) 
    wstar = svm_linear(pca_DTR, LTR, C, K)
    svm_scoring(pca_DTE, LTE, wstar, K)
    print() 

    K = 1
    C = 1
    print("K = %d" % K) 
    print("C = %d" % C) 
    wstar = svm_linear(pca_DTR, LTR, C, K)
    svm_scoring(pca_DTE, LTE, wstar, K)
    print()
    
    print("RBF Kernel:")
    K = 5
    C = 0.1
    gamma = 1
    print("K = %d" % K) 
    print("C = %2f" % C)
    print("gamma = %d" % gamma)
    alphastar, z = svm_RBF(DTR, LTR, C, gamma, K)
    svm_RBF_scoring(DTE, LTE, DTR, K, gamma, alphastar, z)
    print()

    print("RBF Kernel:")
    K = 1
    C = 0.5
    gamma = 10
    print("K = %d" % K) 
    print("C = %2f" % C)
    print("gamma = %d" % gamma)
    alphastar, z = svm_RBF(DTR, LTR, C, gamma, K)
    svm_RBF_scoring(DTE, LTE, DTR, K, gamma, alphastar, z)
    print()

    print("RBF Kernel:")
    K = 2
    C = 1
    gamma = 2
    print("K = %d" % K) 
    print("C = %2f" % C)
    print("gamma = %d" % gamma)
    alphastar, z = svm_RBF(DTR, LTR, C, gamma, K)
    svm_RBF_scoring(DTE, LTE, DTR, K, gamma, alphastar, z)
    print()
    '''

    print("Poly Kernel:")
    K = 3
    C = 0.1
    c = 1
    d = 2
    print("K = %d" % K) 
    print("C = %2f" % C)
    print("c = %2f" % c)
    print("d = %2f" % d)
    alphastar, z = svm_poly(DTR, LTR, C, c, d, K)
    svm_poly_scoring(DTE, LTE, DTR, K, c, d, alphastar, z)
    print()
    
    print("/////////////////////////////////////////////////////////////")    

    '''
    gaussian mixture models classifier results
    '''
    print()
    print("GMM with raw data:")
    GMM_classifier(DTR, LTR, DTE, LTE, "raw data")
    print()
    print("GMM with gaussianized data:")
    GMM_classifier(DTR_g, LTR, DTE_g, LTE, "gaussianized data")
    print()
    print("GMM with PCA with components = 10 and raw data:")
    GMM_classifier(pca_DTR, LTR, pca_DTE, LTE, "PCA data")
    print()
    print("GMM with PCA with components = 5 and raw data:")
    GMM_classifier(pca_DTR5, LTR, pca_DTE5, LTE, "PCA data - m=5")
    print()
    print("GMM with PCA with components = 10 and gaussianized data:")
    GMM_classifier(pca_DTR_g, LTR, pca_DTE_g, LTE, "PCA and gaussianized data")
    print()
    print("/////////////////////////////////////////////////////////////")

    '''
    gaussian mixture models naive bayes classifier results
    '''
    print()
    print("GMM naive Bayes with raw data:")
    GMM_naivebayes_classifier(DTR, LTR, DTE, LTE, "raw data")
    print()
    print("GMM naive Bayes with gaussianized data:")
    GMM_naivebayes_classifier(DTR_g, LTR, DTE_g, LTE, "gaussianized data")
    print()
    print("GMM naive Bayes with PCA with components = 10 and raw data:")
    GMM_naivebayes_classifier(pca_DTR, LTR, pca_DTE, LTE, "PCA data")
    print()
    print("GMM naive Bayes with PCA with components = 5 and raw data:")
    GMM_naivebayes_classifier(pca_DTR5, LTR, pca_DTE5, LTE, "PCA data")
    print()
    print("GMM naive Bayes with PCA with components = 10 and gaussianized data:")
    GMM_naivebayes_classifier(pca_DTR_g, LTR, pca_DTE_g, LTE, "PCA and gaussianized data")
    print()
    print("/////////////////////////////////////////////////////////////")

    '''
    gaussian mixture models naive bayes classifier results
    '''
    print()
    print("GMM tied covariance with raw data:")
    GMM_tiedcov_classifier(DTR, LTR, DTE, LTE, "raw data")
    print()
    print("GMM tied covariance with gaussianized data:")
    GMM_tiedcov_classifier(DTR_g, LTR, DTE_g, LTE, "gaussianized data")
    print()
    print("GMM tied covariance with PCA with components = 10 and raw data:")
    GMM_tiedcov_classifier(pca_DTR, LTR, pca_DTE, LTE, "PCA data")
    print()
    print("GMM tied covariance with PCA with components = 5 and raw data:")
    GMM_tiedcov_classifier(pca_DTR5, LTR, pca_DTE5, LTE, "PCA data")
    print()
    print("GMM tied covariance with PCA with components = 10 and gaussianized data:")
    GMM_tiedcov_classifier(pca_DTR_g, LTR, pca_DTE_g, LTE, "PCA and gaussianized data")
    print()
    print("/////////////////////////////////////////////////////////////")
