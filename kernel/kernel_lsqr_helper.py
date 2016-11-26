import os
import sys
import numpy as np
from params import Param
from sklearn import svm,preprocessing,cross_validation
from model_def import ModelInf
import math
import scipy
import sklearn.metrics as metrics
import gc

def block_kernel_solve(K, y, numiter=1, block_size=4000,num_classes=10, epochs=3, lambdav=0.1, verbose=True,val_K=None,val_y=None):
        '''Solve (K + \lambdaI)x = y
            in a block-wise fashion
        '''

        # compute some constants
        num_samples = K.shape[0]
        num_blocks = math.ceil(num_samples*1.0/block_size)
        x = np.zeros((K.shape[0], num_classes))
        y_hat = np.zeros((K.shape[0], num_classes))
        onehot = lambda x: np.eye(num_classes)[x]
        y_onehot = np.array(map(onehot, y))
        loss = 0
        print num_blocks
        idxes = np.diag_indices(num_samples)
        if num_blocks==1:
            epochs=1

        for e in range(epochs):
                shuffled_coords = np.random.choice(num_samples, num_samples, replace=False)
                for b in range(int(num_blocks)):
                        # pick a block
                        K[idxes] += lambdav
                        block = shuffled_coords[b*block_size:min((b+1)*block_size, num_samples)]

                        # pick a subset of the kernel matrix (note K can be mmap-ed)
                        K_block = K[:, block]

                        # This is a matrix vector multiply very efficient can be parallelized
                        # (even if K is mmaped)

                        # calculate
                        residuals = y_onehot - y_hat


                        # should be block size x block size
                        KbTKb = K_block.T.dot(K_block)

                        print("solving block {0}".format(b))
                        try:
                            x_block = scipy.linalg.solve(KbTKb, K_block.T.dot(residuals))
                        except:
                            return None


                        # update model
                        x[block] = x[block]+x_block
                        K[idxes] -= lambdav
                        y_hat = K.dot(x)

                        y_pred = np.argmax(y_hat, axis=1)
                        train_acc = metrics.accuracy_score(y, y_pred)
                        if (verbose):
                                print "Epoch: {0}, Block: {2}, Loss: {3}, Train Accuracy: {1}".format(e, train_acc, b, loss)
                if val_K is not None:
                    val_hat = val_K.dot(x)
                    val_pred = np.argmax(val_hat, axis=1)
                    val_acc = metrics.accuracy_score(val_y, val_pred)
                    if (verbose):
                            print "Epoch: {0}, Val Accuracy: {1}".format(e, val_acc)
        return x

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



class svm_model(ModelInf):
    def __init__(self,name, data_dir, seed,combine=False):
        self.data_dir=data_dir
        os.chdir(data_dir)
        self.name=name
        self.data=None
        self.orig_data=create_dataset(name,data_dir,combine)
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
    def compute_preprocessor(self,method):
        self.data={}
        if method=='min_max':
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
    def run_solver(self, unit, n_units, arm,solver_type='lsqr'):
        kernel_map=dict(zip([1,2,3],['rbf','poly','sigmoid']))
        preprocess_map=dict(zip([1,2,3],['min_max','scaled','normalized']))
        self.compute_preprocessor(preprocess_map[arm['preprocessor']])
        print arm
        # Shuffle the data and split up the request subset of the training data
        size = int(n_units)
        s_max = self.data['y_train'].shape[0]
        if n_units<s_max:
            shuffle = cross_validation.StratifiedShuffleSplit(self.data['y_train'],3,test_size=n_units,random_state=0)
            for train_index, test_index in shuffle:
                train_subset = self.data['X_train'][test_index]
                train_targets_subset = self.data['y_train'][test_index]
        else:
            train_subset=self.data['X_train']
            train_targets_subset=self.data['y_train']
        # Train model on the subset
        if solver_type=='SVM':
            clf = svm.SVC(C=arm['C'], kernel=kernel_map[arm['kernel']], gamma=arm['gamma'], coef0=arm['coef0'], degree=arm['degree'])
            clf.fit(train_subset, train_targets_subset)

            # Validate this hyperparameter configuration on the full validation data
            #y_loss = 1 - clf.score(self.data['X_train'], self.data['y_train'])
            y_loss=1
            test_acc=0
            val_acc= clf.score(self.data['X_val'], self.data['y_val'])
            if n_units==s_max:
                test_acc = clf.score(self.data['X_test'], self.data['y_test'])
        else:
            kernel_type=kernel_map[arm['kernel']]
            if kernel_type=='rbf':
                K=metrics.pairwise.pairwise_kernels(train_subset,metric=kernel_map[arm['kernel']], gamma=arm['gamma'])
                val_kernel=metrics.pairwise.pairwise_kernels(self.data['X_val'],train_subset,metric=kernel_map[arm['kernel']], gamma=arm['gamma'])
                if n_units==s_max:
                    test_kernel=metrics.pairwise.pairwise_kernels(self.data['X_test'],train_subset,metric=kernel_map[arm['kernel']], gamma=arm['gamma'])
            elif kernel_type=='poly':
                K=metrics.pairwise.pairwise_kernels(train_subset,metric=kernel_map[arm['kernel']], gamma=arm['gamma'],degree=arm['degree'],coef0=arm['coef0'])
                val_kernel=metrics.pairwise.pairwise_kernels(self.data['X_val'],train_subset,metric=kernel_map[arm['kernel']], gamma=arm['gamma'],degree=arm['degree'],coef0=arm['coef0'])
                if n_units==s_max:
                    test_kernel=metrics.pairwise.pairwise_kernels(self.data['X_test'],train_subset,metric=kernel_map[arm['kernel']], gamma=arm['gamma'],degree=arm['degree'],coef0=arm['coef0'])
            elif kernel_type=='sigmoid':
                K=metrics.pairwise.pairwise_kernels(train_subset,metric=kernel_map[arm['kernel']], gamma=arm['gamma'],coef0=arm['coef0'])
                val_kernel=metrics.pairwise.pairwise_kernels(self.data['X_val'],train_subset,metric=kernel_map[arm['kernel']], gamma=arm['gamma'],coef0=arm['coef0'])
                if n_units==s_max:
                    test_kernel=metrics.pairwise.pairwise_kernels(self.data['X_test'],train_subset,metric=kernel_map[arm['kernel']], gamma=arm['gamma'],coef0=arm['coef0'])

            x=block_kernel_solve(K,train_targets_subset,lambdav=1/arm['C']*(n_units))
            if x is None:
                return 1, 0 ,0
            y_loss=1
            test_acc=0
            y_pred=np.argmax(val_kernel.dot(x),axis=1)
            val_acc=metrics.accuracy_score(y_pred,self.data['y_val'])
            if n_units==s_max:
                y_pred=np.argmax(test_kernel.dot(x),axis=1)
                test_acc=metrics.accuracy_score(y_pred,self.data['y_test'])
                del test_kernel
            del K,val_kernel
        del self.data
        gc.collect()

        return y_loss,val_acc,test_acc

def get_svm_search():
    params = {}
    params['C']=Param('C',-3.0,5.0,distrib='uniform',scale='log',logbase=10.0)
    params['gamma']=Param('gamma',-5.0,1.0,distrib='uniform',scale='log',logbase=10.0)
    params['kernel']=Param('kernel',1,4,distrib='uniform',scale='linear',interval=1)
    params['preprocessor']=Param('kernel',1,4,distrib='uniform',scale='linear',interval=1)
    params['coef0']=Param('coef0',-1.0,1.0,distrib='uniform',scale='linear')
    params['degree']=Param('degree',2,6,distrib='uniform',scale='linear',interval=1)

    return params
def main():
    # Use for testing.
    data_dir="/home/lisha/school/Projects/hyperband_nnet/hyperband2/svm/cifar10"
    model=svm_model("cifar10",data_dir,3000,False)
    arm={}
    arm['dir']=data_dir+"/hyperband_constant/trial2000/default_arm"
    arm['kernel']=1
    arm['C']=64286.7665
    arm['degree']=2
    arm['coef0']=0.29041
    arm['gamma']=0.006859435
    arm['preprocessor']=1
    arm['results']=[]
    train_loss,val_acc,test_acc=model.run_solver('iter',40000,arm,solver_type='lsqr')
    print train_loss, test_acc
if __name__ == "__main__":
    main()
