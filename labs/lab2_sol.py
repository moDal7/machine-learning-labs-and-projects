import numpy
import matplotlib
import matplotlib.pyplot as plt


# function to reshape vectors 
def mcol(v):
    return v.reshape((v.size, 1))

def load(fname):
    DList=[]
    labelsList=[]

    # dictionary for labels
    hlabels={
        'iris-setosa': 0,
        'iris-versicolor': 1,
        'iris-virginica': 2
        }
    
    # load function, builds attributes numpy array and labels one
    with open(fname) as f:
        for line in f:
            try:
                attrs=line.split(',')[0:4]
                attrs=mcol(numpy.array([float(i) for i in attrs]))
                name = line.split(',')[-1].strip()
                label=hlabels(name)
                DList.append(attrs)
                labelsList.append(label)
            except:
                pass

    return numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)

def load2():

    # The dataset is already available in the sklearn library (pay attention that the library represents samples as row vectors, not column vectors - we need to transpose the data matrix)
    import sklearn.datasets
    return sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
