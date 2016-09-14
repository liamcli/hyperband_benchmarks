import time
import numpy
import pickle
import os
import sys,getopt
from params import zoom_space



class Logger(object):
    def __init__(self,dir):
        self.terminal = sys.stdout
        self.log = open(dir+"/random_run.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
def random_search(model,runtime,units,dir,params,max_units):
    # input t in minutes
    t_0 = time.time()
    print time.localtime(t_0)
    def minutes(t):
        return (t-t_0)/60.
    results=[]
    while minutes(time.time())< runtime:
        start_time=time.time()
        arm = model.generate_arms(1,dir,params)
        train_loss,val_acc,test_acc=model.run_solver(units, max_units,arm[0])
        run_time=(time.time()-start_time)/60.0
        print train_loss, val_acc, test_acc, run_time
        results.append([train_loss,val_acc,test_acc,run_time])
    pickle.dump(results,open('results.pkl','wb'))

def main(argv):

    model=''
    data_dir=''
    output_dir=''
    seed_id=0
    device_id=0
    try:
        opts, args = getopt.getopt(argv,"hm:i:o:R:s:d:",['model=','input_dir=','output_dir=','max_iter=','seed=','device='])
    except getopt.GetoptError:
        print 'random_search.py -m <model> -i <data_dir> -o <output_dir> -R <max_iter> -s <rng_seed> -d <GPU_id>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'random_search.py -i <data_dir> -o <output_dir> -s <rng_seed> -d <GPU_id>'
            sys.exit()
        elif opt in ("-m", "--model"):
            model = arg
        elif opt in ("-i", "--input_dir"):
            data_dir = arg
        elif opt in ("-o", "--output_dir"):
            output_dir = arg
        elif opt in ("-R", "--max_iter"):
            max_units = int(arg)
        elif opt in ("-s", "--seed"):
            seed_id = int(arg)
        elif opt in ("-d", "--device"):
            device_id= int(arg)
    dir=output_dir+'/trial'+str(seed_id)
    #Starting 6 used increasing budget, before used constant budget for max metaarms
    if not os.path.exists(dir):
        os.makedirs(dir)
    sys.stdout = Logger(dir)
    if model=='cifar10':
        from cifar10.cifar10_helper import get_cnn_search_space,cifar10_conv
        params = get_cnn_search_space()
        obj=cifar10_conv(data_dir,device=device_id,seed=seed_id)
        random_search(obj,360,'iter',dir,params,30000)
    elif model=='svhn':
        from svhn.svhn_helper import get_cnn_search_space,svhn_conv
        params = get_cnn_search_space()
        obj=svhn_conv(data_dir,device=device_id,seed=seed_id)
        random_search(obj,720,'iter',dir,params,60000)
    elif model=='mrbi':
        from mrbi.mrbi_helper import get_cnn_search_space,mrbi_conv
        params = get_cnn_search_space()
        obj=mrbi_conv(data_dir,device=device_id,seed=seed_id)
        random_search(obj,360,'iter',dir,params,30000)
    elif model=='cifar100':
        from networkin.nin_helper import get_nin_search_space,nin_conv
        params = get_nin_search_space()
        obj=nin_conv("cifar100",data_dir,device_id,seed_id)
        random_search(obj,2000,'iter',dir,params,60000)
    elif model=='svm':
        from svm.svm_helper import get_svm_search,svm_model
        params= get_svm_search()
        obj=svm_model('svm',data_dir,seed_id)
        random_search(obj,24*60,'iter',dir,params,40000)
    elif model=='cifar10_random_features':
        from svm.random_features_helper import get_svm_search,random_features_model
        params = get_svm_search()
        obj=random_features_model('cifar10',data_dir,seed=seed_id)
        random_search(obj,12*60,'iters',dir,params,max_units)

if __name__ == "__main__":
    main(sys.argv[1:])