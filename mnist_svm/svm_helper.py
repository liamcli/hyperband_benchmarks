import os
import sys
import numpy as np
from params import Param
from sklearn import svm
from model_def import ModelInf

def load_dataset():
    # This function loads the MNIST data, its copied from the Lasagne tutorial
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    data={}
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    X_train = X_train.reshape(X_train.shape[0], 28 * 28)
    X_val = X_val.reshape(X_val.shape[0], 28 * 28)
    X_test = X_test.reshape(X_test.shape[0], 28 * 28)

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    data['X_train']=X_train
    data['X_test']=X_test
    data['X_val']=X_val
    data['y_train']=y_train
    data['y_test']=y_test
    data['y_val']=y_val
    return data


# The optimization function that we want to optimize.
# It gets a numpy array x with shape (1,D) where D are the number of parameters
# and s which is the ratio of the training data that is used to
# evaluate this configuration
class svm_model(ModelInf):
    def __init__(self,name, data_dir, seed):
        self.data_dir=data_dir
        os.chdir(data_dir)
        self.name="cifar10_conv"
        self.data=load_dataset()
        np.random.seed(seed)
    def generate_arms(self,n,dir, params):
        os.chdir(dir)
        arms={}
        subdirs=next(os.walk('.'))[1]
        if len(subdirs)==0:
            start_count=0
        else:
            start_count=len(subdirs)
        for i in range(n):
            dirname="arm"+str(start_count+i)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            arm={}
            arm['dir']=dir+"/"+dirname
            for hp in ['C','gamma']:
                val=params[hp].get_param_range(1,stochastic=True)
                arm[hp]=val[0]
            arm['results']=[]
            arms[i]=arm
        return arms

    def run_solver(self, unit, n_units, arm):
        # Shuffle the data and split up the request subset of the training data
        size = int(n_units)
        s_max = self.data['y_train'].shape[0]
        shuffle = np.random.permutation(np.arange(s_max))
        train_subset = self.data['X_train'][shuffle[:size]]
        train_targets_subset = self.data['y_train'][shuffle[:size]]

        # Train the SVM on the subset set
        C = arm['C']
        gamma = arm['gamma']
        clf = svm.SVC(gamma=gamma, C=C)
        clf.fit(train_subset, train_targets_subset)

        # Validate this hyperparameter configuration on the full validation data
        y_loss = 1 - clf.score(self.data['X_train'], self.data['y_train'])
        val_acc= clf.score(self.data['X_val'], self.data['y_val'])
        test_acc = clf.score(self.data['X_test'], self.data['y_test'])


        return y_loss,val_acc,test_acc

def get_svm_search():
    params = {}
    params['C']=Param('C',-10.0,10.0,distrib='uniform',scale='log',logbase=np.exp(1))
    params['gamma']=Param('gamma',-10.0,10.0,distrib='uniform',scale='log',logbase=np.exp(1))
    return params

