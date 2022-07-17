import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats as ss
import scipy.special


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

def mcol(v):
    return v.reshape((v.size, 1))

def mrow(v):
    return v.reshape((1, v.size))

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

def eda(data, labels, feature_names):
    mean = compute_mean(data)
    mean_0 = compute_mean(data[:, labels[0, :]==0])
    mean_1 = compute_mean(data[:, labels[0, :]==1])
    cov = compute_cov(data)
    skew = ss.skew(data)

    for i in range(len(feature_names)):
        print()
        print("Feature %d, %s has the following properties:" % (i+1, feature_names[i]))
        print()
        print("Mean is %3f" % (mean[i]))
        print("Mean computed for class %d, is %3f" % (0, mean_0[i]))
        print("Mean computed for class %d, is %3f" % (1, mean_1[i]))
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

def plot_hist(DTR, LTR, bool_save):
    DTR_0 = DTR[:, LTR[0, :]==0]
    DTR_1 = DTR[:, LTR[0, :]==1]
    for i in range(DTR.shape[0]):
        plt.figure()
        plt.xlabel("Attribute %d" % (i))
        plt.ylabel("Frequency")
        plt.hist(DTR_0[i, :], 10, label = 'Class 0', histtype='stepfilled', alpha=0.5, linewidth=1)
        plt.hist(DTR_1[i, :], 10, label = 'Class 1', histtype='stepfilled', alpha=0.5, linewidth=1)
        plt.legend()
        plt.tight_layout()
        plt.show()
        if bool_save:
            plt.savefig('./project/graphs/hist_%d.pdf' % i)

def plot_boxplot(DTR, LTR, bool_save):
    DTR_0 = DTR[:, LTR[0, :]==0]
    DTR_1 = DTR[:, LTR[0, :]==1]
    for i in range(DTR.shape[0]):
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))

        ax1.boxplot(DTR_0[i, :],
                    vert=True,  # vertical box alignment
                    patch_artist=True,  # fill with color
                    )  # will be used to label x-ticks

        ax2.boxplot(DTR_1[i, :],
                    vert=True,  # vertical box alignment
                    patch_artist=True,  # fill with color
                    )  # will be used to label x-ticks
        ax1.set_title('Class 0')
        ax2.set_title('Class 1')
        for ax in [ax1, ax2]:
            ax.yaxis.grid(True)
            ax.set_xlabel('Attribute %d' % (i))
            ax.set_ylabel('Observed values')
        plt.legend()
        plt.tight_layout()
        plt.show()
        if bool_save:
            plt.savefig('./project/graphs/boxplot_%d.pdf' % i)

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
    
    priors = [0, 0]
    for i in range(len(lab)):
       priors[i] = (LTR[0, :]==i).sum(0)/LTR.shape[1]

    log_j_scores = np.empty([len(lab), DTE.shape[1]])
    log_j_scores[0, :] = log_scores[0, :] + np.log(priors[0])
    log_j_scores[1, :] = log_scores[1, :] + np.log(priors[1])

    logSMarginal = scipy.special.logsumexp(log_j_scores, axis=0)

    log_post1 = np.exp(log_j_scores - logSMarginal)
    LPred = log_post1.argmax(0)
    # compare_j_scores = np.load('labs/lab5_results/SJoint_MVG.npy')
    print("Accuracy: %f" % ((LTE[0, :]==LPred).sum(0)/LTE.shape[1])) 

def naive_multivariate_gaussian_classifier(DTR, LTR, DTE, LTE):

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
    
    priors = [0, 0]
    for i in range(len(lab)):
       priors[i] = (LTR[0, :]==i).sum(0)/LTR.shape[1]

    log_j_scores = np.empty([len(lab), DTE.shape[1]])
    log_j_scores[0, :] = log_scores[0, :] + np.log(priors[0])
    log_j_scores[1, :] = log_scores[1, :] + np.log(priors[1])

    logSMarginal = scipy.special.logsumexp(log_j_scores, axis=0)

    log_post1 = np.exp(log_j_scores - logSMarginal)
    LPred = log_post1.argmax(0)
    # compare_j_scores = np.load('labs/lab5_results/SJoint_MVG.npy')
    print("Accuracy: %f" % ((LTE[0, :]==LPred).sum(0)/LTE.shape[1])) 

def tiedcov_multivariate_gaussian_classifier(DTR, LTR, DTE, LTE):

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
    
    priors = [0, 0]
    for i in range(len(lab)):
       priors[i] = (LTR[0, :]==i).sum(0)/LTR.shape[1]

    log_j_scores = np.empty([len(lab), DTE.shape[1]])
    log_j_scores[0, :] = log_scores[0, :] + np.log(priors[0])
    log_j_scores[1, :] = log_scores[1, :] + np.log(priors[1])

    logSMarginal = scipy.special.logsumexp(log_j_scores, axis=0)

    log_post1 = np.exp(log_j_scores - logSMarginal)
    LPred = log_post1.argmax(0)
    # compare_j_scores = np.load('labs/lab5_results/SJoint_MVG.npy')
    print("Accuracy: %f" % ((LTE[0, :]==LPred).sum(0)/LTE.shape[1])) 

def tiednaive_multivariate_gaussian_classifier(DTR, LTR, DTE, LTE):

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
    
    priors = [0, 0]
    for i in range(len(lab)):
       priors[i] = (LTR[0, :]==i).sum(0)/LTR.shape[1]

    log_j_scores = np.empty([len(lab), DTE.shape[1]])
    log_j_scores[0, :] = log_scores[0, :] + np.log(priors[0])
    log_j_scores[1, :] = log_scores[1, :] + np.log(priors[1])

    logSMarginal = scipy.special.logsumexp(log_j_scores, axis=0)

    log_post1 = np.exp(log_j_scores - logSMarginal)
    LPred = log_post1.argmax(0)
    # compare_j_scores = np.load('labs/lab5_results/SJoint_MVG.npy')
    print("Accuracy: %f" % ((LTE[0, :]==LPred).sum(0)/LTE.shape[1])) 

def gaussianization(data):
    # rank computation 
    rank_matrix = np.empty([data.shape[0], data.shape[1]])

    for i in range(data.shape[0]):
        rank_matrix[i, :] = ss.rankdata(data[i, :])
    
    rank_matrix = (rank_matrix)/(data.shape[1]+2)
    
    # ppf function from scipy.stats module 
    gaussianized_data = norm.ppf(rank_matrix)

    return gaussianized_data


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

    
    '''
    multivariate gaussian classifier results
    '''
    print()
    print("MVG with raw data:")
    multivariate_gaussian_classifier(DTR, LTR, DTE, LTE)
    print()
    print("MVG with gaussianized data:")
    multivariate_gaussian_classifier(DTR_g, LTR, DTE_g, LTE)
    print()
    print("MVG with PCA with components = 10 and raw data:")
    multivariate_gaussian_classifier(pca_DTR, LTR, pca_DTE, LTE)
    print()
    print("MVG with PCA with components = 10 and gaussianized data:")
    multivariate_gaussian_classifier(pca_DTR_g, LTR, pca_DTE_g, LTE)
    print()
    print("/////////////////////////////////////////////////////////////")

    '''
    naive bayes multivariate gaussian classifier results
    '''
    print()
    print("Naive Bayes MVG with raw data:")
    naive_multivariate_gaussian_classifier(DTR, LTR, DTE, LTE)
    print()
    print("Naive Bayes MVG with gaussianized data:")
    naive_multivariate_gaussian_classifier(DTR_g, LTR, DTE_g, LTE)
    print()
    print("Naive Bayes MVG with PCA with components = 10 and raw data:")
    naive_multivariate_gaussian_classifier(pca_DTR, LTR, pca_DTE, LTE)
    print()
    print("Naive Bayes MVG with PCA with components = 10 and gaussianized data:")
    naive_multivariate_gaussian_classifier(pca_DTR_g, LTR, pca_DTE_g, LTE)
    print()
    print("/////////////////////////////////////////////////////////////")

    '''
    tied covariance multivariate gaussian classifier results
    '''
    print()
    print("Tied covariance MVG with raw data:")
    tiedcov_multivariate_gaussian_classifier(DTR, LTR, DTE, LTE)
    print()
    print("Tied Covariance MVG with gaussianized data:")
    tiedcov_multivariate_gaussian_classifier(DTR_g, LTR, DTE_g, LTE)
    print()
    print("Tied Covariance MVG with PCA with components = 10 and raw data:")
    tiedcov_multivariate_gaussian_classifier(pca_DTR, LTR, pca_DTE, LTE)
    print()
    print("Tied Covariance MVG with PCA with components = 10 and gaussianized data:")
    tiedcov_multivariate_gaussian_classifier(pca_DTR_g, LTR, pca_DTE_g, LTE)
    print()
    print("/////////////////////////////////////////////////////////////")

    '''
    tied naive covariance multivariate gaussian classifier results
    '''
    print()
    print("Tied Naive covariance MVG with raw data:")
    tiednaive_multivariate_gaussian_classifier(DTR, LTR, DTE, LTE)
    print()
    print("Tied Naive Covariance MVG with gaussianized data:")
    tiednaive_multivariate_gaussian_classifier(DTR_g, LTR, DTE_g, LTE)
    print()
    print("Tied Naive Covariance MVG with PCA with components = 10 and raw data:")
    tiednaive_multivariate_gaussian_classifier(pca_DTR, LTR, pca_DTE, LTE)
    print()
    print("Tied Naive Covariance MVG with PCA with components = 10 and gaussianized data:")
    tiednaive_multivariate_gaussian_classifier(pca_DTR_g, LTR, pca_DTE_g, LTE)
    print()
    print("/////////////////////////////////////////////////////////////")

    # compute_dcf(LPred, LTE)
    # roc curve
    # bayes diagram
    # logistic_regression(DTR, LTR, DTE, LTE)
    # support_vector_machines(DTR, LTR, DTE, LTE)
    # guassian_mixture_models(DTR, LTR, DTE, LTE)
    # commenting hard
    # write the report ehy ?
