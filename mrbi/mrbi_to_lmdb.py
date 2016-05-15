import numpy as np
import lmdb
import caffe
import scipy.io
from PIL import Image
from sklearn.cross_validation import StratifiedShuffleSplit
from matplotlib import pyplot as plt
def parseline(line):
    data = np.array([float(i) for i in line.split()])
    x=data[:-1].reshape((28,28),order='F')
    x=np.array(x*255,dtype=np.uint8)
    x=x[np.newaxis,:,:]
    y=data[-1]
    return x,y
def get_data(filename):
    file = open(filename)

    # Add the lines of the file into a list
    X=[]
    Y=[]
    for line in file:
        x,y=parseline(line)
        X.append(x)
        Y.append(y)
    file.close()
    X=np.array(X)
    Y=np.array(Y,dtype=int)
    return X,Y
def create_datum(x,y):
    datum=caffe.proto.caffe_pb2.Datum()
    datum.data=x.tostring()
    datum.label=y
    return datum
def make_test():
    print 'Loading Matlab data.'
    f = '/home/lisha/school/Projects/hyperband_nnet/hyperband2/mrbi/mnist_rotation_back_image_new/mnist_all_background_images_rotation_normalized_test.amat'

    # name of your matlab variables:

    X,Y=get_data(f)
    N = Y.shape[0]
    map_size = X.nbytes*2
    #if you want to shuffle your data
    #random.shuffle(N)
    env = lmdb.open('/home/lisha/school/Projects/hyperband_nnet/hyperband2/mrbi/mrbi_test', map_size=map_size)
    with env.begin(write=True) as txn:
        # txn is a Transaction object
        for i in range(N):
            im_dat = caffe.io.array_to_datum(X[i],Y[i])
            txn.put('{:0>10d}'.format(i), im_dat.SerializeToString())

def make_train_val():
    print 'Loading Matlab data.'
    f =  '/home/lisha/school/Projects/hyperband_nnet/hyperband2/mrbi/mnist_rotation_back_image_new/mnist_all_background_images_rotation_normalized_train_valid.amat'

    X,Y=get_data(f)
    N = Y.shape[0]
    map_size = X.nbytes*2
    #if you want to shuffle your data
    #random.shuffle(N)

    sss = StratifiedShuffleSplit(Y, 3, test_size=2000, random_state=0)
    for train_index, test_index in sss:
        ind_train1=train_index
        ind_val1=test_index
    print len(ind_train1),len(ind_val1)
    env = lmdb.open('/home/lisha/school/Projects/hyperband_nnet/hyperband2/mrbi/mrbi_train', map_size=map_size*5/6)
    with env.begin(write=True) as txn:
        # txn is a Transaction object
        for i in range(len(ind_train1)):
            im_dat = caffe.io.array_to_datum(X[ind_train1[i]],Y[ind_train1[i]])
            txn.put('{:0>10d}'.format(i), im_dat.SerializeToString())
    env = lmdb.open('/home/lisha/school/Projects/hyperband_nnet/hyperband2/mrbi/mrbi_val', map_size=map_size/6)
    with env.begin(write=True) as txn:
        # txn is a Transaction object
        for i in range(len(ind_val1)):
            im_dat = caffe.io.array_to_datum(X[ind_val1[i]],Y[ind_val1[i]])
            txn.put('{:0>10d}'.format(i), im_dat.SerializeToString())


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