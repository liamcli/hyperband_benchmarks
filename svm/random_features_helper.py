import os
import sys
import numpy as np
from params import Param
from sklearn import linear_model,preprocessing,kernel_approximation
from model_def import ModelInf
import time
import gc

def create_dataset(data_name,data_dir,combine=False):
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
        batch=pickle.load(open(data_dir+'/cifar-10-batches-py/data_batch_5','rb'))
        X_val=np.array(batch['data'],dtype=np.float32)
        y_val=np.array(batch['labels'])
        batch=pickle.load(open(data_dir+'/cifar-10-batches-py/test_batch','rb'))
        X_test=np.array(batch['data'],dtype=np.float32)
        y_test=np.array(batch['labels'])
        if combine:
            X_train=np.concatenate((X_val,X_train))
            y_train=np.concatenate((y_val,y_train))



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
class random_features_model(ModelInf):
    def __init__(self,name, data_dir, seed,combine=False):
        self.data_dir=data_dir
        os.chdir(data_dir)
        self.name=name
        self.data=None
        self.orig_data=create_dataset(name,data_dir,combine)
        np.random.seed(seed)


    def generate_arms(self,n,dir, params,max_iter=None):
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
    def compute_preprocessor(self,method):
        self.data={}
        if method=='none':
            self.data=self.orig_data
        elif method=='min_max':
            transform=preprocessing.MinMaxScaler()
            self.data['X_train']=transform.fit_transform(self.orig_data['X_train'])
            self.data['X_val']=transform.transform(self.orig_data['X_val'])
            self.data['X_test']=transform.transform(self.orig_data['X_test'])
        elif method=='scaled':
            self.data['X_train']=preprocessing.scale(self.orig_data['X_train'])
            self.data['X_val']=preprocessing.scale(self.orig_data['X_val'])
            self.data['X_test']=preprocessing.scale(self.orig_data['X_test'])
        elif method=='normalized':
            self.data['X_train']=preprocessing.normalize(self.orig_data['X_train'])
            self.data['X_val']=preprocessing.normalize(self.orig_data['X_val'])
            self.data['X_test']=preprocessing.normalize(self.orig_data['X_test'])
        self.data['y_train']=self.orig_data['y_train']
        self.data['y_val']=self.orig_data['y_val']
        self.data['y_test']=self.orig_data['y_test']
    def run_solver(self, unit, n_units, arm):
        start_time=time.time()
        #kernel_map=dict(zip([1,2,3],['rbf','poly','sigmoid']))
        preprocess_map=dict(zip([1,2,3,4],['none','min_max','scaled','normalized']))
        self.compute_preprocessor(preprocess_map[arm['preprocessor']])
        print arm
        # Create random features
        features=kernel_approximation.RBFSampler(gamma=arm['gamma'],n_components=n_units, random_state=1)
        train_features=features.fit_transform(self.data['X_train'])
        val_features=features.transform(self.data['X_val'])
        test_features=features.transform(self.data['X_test'])
        approx_time=(time.time()-start_time)/60.0
        print 'approximating kernel took %r' % approx_time

        clf = linear_model.RidgeClassifier(alpha=1.0/(arm['C']*n_units),solver='lsqr',copy_X=False)
        clf.fit(train_features, self.data['y_train'])
        print 'fitting model took %r' % ((time.time()-start_time)/60.0 - approx_time)
        # Validate this hyperparameter configuration on the full validation data
        #y_loss = 1 - clf.score(self.data['X_train'], self.data['y_train'])
        y_loss=1
        test_acc=0
        val_acc= clf.score(val_features, self.data['y_val'])
        test_acc = clf.score(test_features, self.data['y_test'])
        del self.data
        del train_features, val_features, test_features
        gc.collect()

        return y_loss,val_acc,test_acc

def get_svm_search():
    params = {}
    params['C']=Param('C',-10.0,-2.0,distrib='uniform',scale='log',logbase=10.0)
    params['gamma']=Param('gamma',-5.0,1.0,distrib='uniform',scale='log',logbase=10.0)
    params['preprocessor']=Param('kernel',1,5,distrib='uniform',scale='linear',interval=1)

    return params
def main():
    data_dir="/home/lisha/school/Projects/hyperband_nnet/hyperband2/svm/cifar10"
    model=random_features_model("cifar10",data_dir,4000,True)
    arm={}
    arm['dir']=data_dir+"/hyperband_constant/trial1002/default_arm"
    arm['preprocessor']=3
    arm['C']=0.0011215088169
    arm['gamma']=0.0001644725
    arm['results']=[]
    train_loss,val_acc,test_acc=model.run_solver('iter',25000,arm)
    print train_loss, test_acc
if __name__ == "__main__":
    main()
