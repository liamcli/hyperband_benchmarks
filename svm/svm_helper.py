import os
import sys
import numpy as np
from params import Param
from sklearn import svm,preprocessing
from model_def import ModelInf

def load_dataset(data_name,data_dir):
    # This function loads the MNIST data, its copied from the Lasagne tutorial
    # We first define a download function, supporting both Python 2 and 3.
    if data_name=='mnist':
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
    elif data_name=='cifar10':
        import pickle
        for f in range(1,5):
            batch=pickle.load(open(data_dir+'/cifar-10-batches-py/data_batch_'+str(f),'rb'))
            if f==1:
                X_train=np.array(batch['data'],dtype=np.float32)
                y_train=np.array(batch['labels'])
            else:
                X_train=np.concatenate((X_train,np.array(batch['data'],dtype=np.float32)))
                y_train=np.concatenate((y_train,np.array(batch['labels'])))
        X_train=preprocessing.normalize(X_train)
        batch=pickle.load(open(data_dir+'/cifar-10-batches-py/data_batch_5','rb'))
        X_val=np.array(batch['data'],dtype=np.float32)
        X_val=preprocessing.normalize(X_val)
        y_val=np.array(batch['labels'])
        batch=pickle.load(open(data_dir+'/cifar-10-batches-py/test_batch','rb'))
        X_test=np.array(batch['data'],dtype=np.float32)
        X_test=preprocessing.normalize(X_test)
        y_test=np.array(batch['labels'])



    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    data={}
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
        self.data=load_dataset(name,data_dir)
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
            for hp in params.keys():
                val=params[hp].get_param_range(1,stochastic=True)
                arm[hp]=val[0]
            arm['results']=[]
            arms[i]=arm
        return arms

    def run_solver(self, unit, n_units, arm):
        kernel_map=dict(zip([1,2,3,4],['linear','rbf','poly','sigmoid']))
        print arm
        # Shuffle the data and split up the request subset of the training data
        size = int(n_units)
        s_max = self.data['y_train'].shape[0]
        shuffle = np.random.permutation(np.arange(s_max))
        train_subset = self.data['X_train'][shuffle[:size]]
        train_targets_subset = self.data['y_train'][shuffle[:size]]

        # Train the SVM on the subset set
        clf = svm.SVC(C=arm['C'], kernel=kernel_map[arm['kernel']], gamma=arm['gamma'], coef0=arm['coef0'], degree=arm['degree'])
        clf.fit(train_subset, train_targets_subset)

        # Validate this hyperparameter configuration on the full validation data
        #y_loss = 1 - clf.score(self.data['X_train'], self.data['y_train'])
        y_loss=1
        test_acc=0
        val_acc= clf.score(self.data['X_val'], self.data['y_val'])
        if n_units==s_max:
            test_acc = clf.score(self.data['X_test'], self.data['y_test'])


        return y_loss,val_acc,test_acc

def get_svm_search():
    params = {}
    params['C']=Param('C',-5.0,5.0,distrib='uniform',scale='log',logbase=10.0)
    params['gamma']=Param('gamma',-5.0,5.0,distrib='uniform',scale='log',logbase=10.0)
    params['kernel']=Param('kernel',1,5,distrib='uniform',scale='linear',interval=1)
    params['coef0']=Param('coef0',-1.0,1.0,distrib='uniform',scale='linear')
    params['degree']=Param('degree',1,6,distrib='uniform',scale='linear',interval=1)

    return params

