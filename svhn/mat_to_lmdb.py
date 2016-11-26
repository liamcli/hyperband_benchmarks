import numpy as np
import lmdb
import caffe
import scipy.io
from PIL import Image
from sklearn.cross_validation import StratifiedShuffleSplit
from matplotlib import pyplot as plt
def make_test():
    print 'Loading Matlab data.'
    f = scipy.io.loadmat('/home/lisha/school/Projects/hyperband_nnet/hyperband2/svhn/svhn_data/test_32x32.mat')
    # name of your matlab variables:
    data = f.get('X')
    labels = f.get('y')

    print 'Creating label dataset.'
    Y = np.array(labels,dtype=int)
    Y[Y==10]=0
    Y= Y.flatten()
    #Y = np.array(Y,dtype=np.float32)
    map_size = Y.nbytes*2
    N = Y.shape[0]
    X = np.array(data)
    X=np.rollaxis(X,3,0)
    map_size = X.nbytes*2
    #if you want to shuffle your data
    #random.shuffle(N)
    env = lmdb.open('/home/lisha/school/Projects/hyperband_nnet/hyperband2/svhn/svhn_test', map_size=map_size)
    with env.begin(write=True) as txn:
        # txn is a Transaction object
        for i in range(N):
            im_dat = caffe.io.array_to_datum(np.rollaxis(X[i],2,0),Y[i])
            txn.put('{:0>10d}'.format(i), im_dat.SerializeToString())

def make_train_val():
    print 'Loading Matlab data.'
    f1 = scipy.io.loadmat('/home/lisha/school/Projects/hyperband_nnet/hyperband2/svhn/svhn_data/train_32x32.mat')
    f2 = scipy.io.loadmat('/home/lisha/school/Projects/hyperband_nnet/hyperband2/svhn/svhn_data/extra_32x32.mat')
    # name of your matlab variables:
    data_train = f1.get('X')
    labels_train = f1.get('y')
    data_extra=f2.get('X')
    labels_extra = f2.get('y')
    sss = StratifiedShuffleSplit(labels_train, 3, test_size=0.05460229056, random_state=0)
    for train_index, test_index in sss:
        ind_train1=train_index
        ind_val1=test_index
    sss = StratifiedShuffleSplit(labels_extra, 3, test_size=0.00376554936, random_state=1)
    for train_index, test_index in sss:
        ind_train2=train_index
        ind_val2=test_index
    print 'val: '+str(len(ind_val1)+len(ind_val2))+' train: '+str(len(ind_train1)+len(ind_train2))
    Y1= np.array(labels_train,dtype=int)
    Y1[Y1==10]=0
    Y1=Y1.flatten()
    Y2= np.array(labels_extra,dtype=int)
    Y2[Y2==10]=0
    Y2=Y2.flatten()

    X1= np.array(data_train)
    X1=np.rollaxis(X1,3,0)

    X2= np.array(data_extra)
    X2=np.rollaxis(X2,3,0)
    map_size_train = X2.nbytes*4
    map_size_val = X1.nbytes*2
    #if you want to shuffle your data
    #random.shuffle(N)
    env = lmdb.open('/home/lisha/school/Projects/hyperband_nnet/hyperband2/svhn/svhn_val', map_size=map_size_val)
    with env.begin(write=True) as txn:
        # txn is a Transaction object
        for i in range(len(ind_val1)):
            im_dat = caffe.io.array_to_datum(np.rollaxis(X1[ind_val1[i]],2,0),Y1[ind_val1[i]])
            txn.put('{:0>10d}'.format(i), im_dat.SerializeToString())
        for i in range(len(ind_val2)):
            im_dat = caffe.io.array_to_datum(np.rollaxis(X2[ind_val2[i]],2,0),Y2[ind_val2[i]])
            txn.put('{:0>10d}'.format(len(ind_val1)+i), im_dat.SerializeToString())
    env = lmdb.open('/home/lisha/school/Projects/hyperband_nnet/hyperband2/svhn/svhn_train', map_size=map_size_train)
    with env.begin(write=True) as txn:
        # txn is a Transaction object
        for i in range(len(ind_train1)):
            im_dat = caffe.io.array_to_datum(np.rollaxis(X1[ind_train1[i]],2,0),Y1[ind_train1[i]])
            txn.put('{:0>10d}'.format(i), im_dat.SerializeToString())
        for i in range(len(ind_train2)):
            im_dat = caffe.io.array_to_datum(np.rollaxis(X2[ind_train2[i]],2,0),Y2[ind_train2[i]])
            txn.put('{:0>10d}'.format(len(ind_train1)+i), im_dat.SerializeToString())

def print_image():
    np.set_printoptions(threshold=np.nan)

    mat = scipy.io.loadmat('/home/lisha/school/Projects/hyperband_nnet/hyperband2/svhn/svhn_data/train_32x32.mat')
    arr =  np.array(mat["X"])

    for i in range(10):
        img =  Image.fromarray(arr[:,:,:,i], 'RGB')
        img.save("svhn"+str(i)+".jpg")
        print "saving: " + str(i)

def view_lmdb_data():
    lmdb_env = lmdb.open('/home/lisha/school/Projects/hyperband_nnet/hyperband2/svhn/svhn_train/')
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe.proto.caffe_pb2.Datum()
    x=[]
    y=[]

    for key, value in lmdb_cursor:
        datum.ParseFromString(value)
        label = datum.label
        data = caffe.io.datum_to_array(datum)
        #plt.imshow(np.rollaxis(data,0,3))
        x.append(data)
        y.append(label)
    print len(y)
def main():
    make_test()
    make_train_val()
    #print_image()
    #view_lmdb_data()

if __name__ == "__main__":
    main()