import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats as ss


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


if __name__ == '__main__':

    # data sources
    train_fname = "./project/data/Train.txt"
    test_fname = "./project/data/Test.txt"
    
    # data and labels loading
    DTR, LTR = load_set(train_fname)
    DTE, LTE = load_set(test_fname)
    feature_names = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", 
                    "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]

    # data gaussianization
    DTR_g = gaussianization(DTR)
    DTE_g = gaussianization(DTE)

    # heat map of correlation matrix
    correlation_matrix(DTR_g, feature_names)

    # data histogram plotting, raw and gaussianized
    plot_hist(DTR, LTR, False)
    # plot_hist(DTR, LTR, True)
    plot_hist(DTR_g, LTR, False)
    # plot_hist(DTR_g, LTR, True)

    # raw data boxplot
    plot_boxplot(DTR, LTR, False)
    # plot_boxplot(DTR, LTR, True)

    # exploratory_data_analysis(DTR, LTR)

    # plot_boxplot(DTR, LTR)
    # plot_scatter(DTR, LTR)
    # pca(DTR, LTR)
    # lda(DTR, LTR)
    # compute_dcf(LPred, LTE)
    # logistic_regression(DTR, LTR, DTE, LTE)
    # support_vector_machines(DTR, LTR, DTE, LTE)
    # guassian_mixture_models(DTR, LTR, DTE, LTE)

    multivariate_gaussian_classifier(DTR, LTR, DTE, LTE)
    multivariate_gaussian_classifier(DTR_g, LTR, DTE_g, LTE)