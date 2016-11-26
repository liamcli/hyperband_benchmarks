import numpy as np
import lmdb
import caffe
from sklearn.cross_validation import StratifiedShuffleSplit

def get_data():
    lmdb_env = lmdb.open('/home/lisha/school/caffe/examples/cifar10/cifar10_train_lmdb//')
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe.proto.caffe_pb2.Datum()

    x=[]
    y=[]
    for key, value in lmdb_cursor:
        datum.ParseFromString(value)
        label = datum.label
        data = caffe.io.datum_to_array(datum)
        x.append(data)
        y.append(label)

    x=np.array(x)
    y=np.array(y)



    map_size = int(1e12)
    # Don't need to shuffle 3 times
    sss = StratifiedShuffleSplit(y, 3, test_size=0.2, random_state=0)
    for train_index, test_index in sss:
        ind_train=train_index
        ind_test=test_index
    env = lmdb.open('/home/lisha/school/caffe/examples/cifar10/cifar10_evenval_lmdb2/', map_size=map_size)
    with env.begin(write=True) as txn:
        # txn is a Transaction object
        for i in range(10000):
            im_dat = caffe.io.array_to_datum(x[ind_test][i],y[ind_test][i])
            txn.put('{:0>10d}'.format(i), im_dat.SerializeToString())

    #map_size = x.nbytes * 10
    del env
    env = lmdb.open('/home/lisha/school/caffe/examples/cifar10/cifar10_eventrain_lmdb2/', map_size=map_size)
    with env.begin(write=True) as txn:
        # txn is a Transaction object
        for i in range(40000):
            im_dat = caffe.io.array_to_datum(x[ind_train][i],y[ind_train][i])
            txn.put('{:0>10d}'.format(i), im_dat.SerializeToString())

def main():
    get_data()
if __name__ == "__main__":
    main()
