import time
import numpy
import pickle
import os
import sys,getopt
from params import zoom_space
from cifar10.cifar10_helper import cifar10_conv,get_cnn_search_space
from svhn.svhn_helper import svhn_conv
from mrbi.mrbi_helper import mrbi_conv
from networkin.nin_helper import nin_conv,get_nin_search_space
from mnist_svm.svm_helper import *


class Logger(object):
    def __init__(self,dir):
        self.terminal = sys.stdout
        self.log = open(dir+"/hyperband_run.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
def hyperband_finite(model,runtime,units,dir,params,min_units,max_units,bounded=True, adaptive=True):
    # input t in minutes
    t_0 = time.time()
    print time.localtime(t_0)
    def minutes(t):
        return (t-t_0)/60.
    k=2
    results_dict={}
    time_test=[]
    while minutes(time.time())< runtime and k <4:

        eta = 4.
        def logeta(x):
            return numpy.log(x)/numpy.log(eta)
        if bounded:
            B=(int(logeta(max_units/min_units))+1)*max_units
        else:
            B = int((2**k)*max_units)
        #B = 15*max_units


        k+=1
        #if bounded:
        #    max_halvings = int(numpy.log(max_units/min_units)/numpy.log(eta))
        #    k = min(numpy.ceil(numpy.log2(max_halvings+1)),k)
        print "\nBudget B = %d" % B
        print '###################'


        # s_max defines the number of inner loops per unique value of B
        # it also specifies the maximum number of rounds
        R = float(max_units)
        r = float(min_units)
        ell_max = int(min(B/R-1,int(logeta(R/r))))
        ell = ell_max
        best_val =0

        while ell >= 0 and minutes(time.time())< runtime:
        #while minutes(time.time())< runtime:

            # specify the number of arms and the number of times each arm is pulled per stage within this innerloop
            n = int( B/R*eta**ell/(ell+1.) )

            if n> 0:
                s = 0
                while (n)*R*(s+1.)*eta**(-s)>B:
                    s+=1
                #s-=1

                print
                print 's=%d, n=%d' %(s,n)
                print 'n_i\tr_k'
                arms,result = sha_finite(model,params,units, n,s,eta,R,dir)
                results_dict[(k,ell)]=arms
                print "k="+str(k)+", l="+str(ell)+", val_acc="+str(result[2])+", test_acc="+str(result[3])+" best_arm_dir: " + result[0]['dir']
                time_test.append([minutes(time.time()),result])
                print "time elapsed: "+ str(minutes(time.time()))
                if result[2]>best_val:
                    best_val=result[2]
                    best_n=n
                    best_s=s
                    best_arm=result[0]

                ell-=1

        #print minutes(time.time())
        #print time.localtime(time.time())
        if adaptive:
            zoom_params=zoom_space(params,best_arm)
            arms,result = sha_finite(model,zoom_params,units, best_n,best_s,eta,R,dir)
            results_dict[(k,ell)]=arms
            print "k="+str(k)+", l="+str(ell)+", val_acc="+str(result[2])+", test_acc="+str(result[3])+" best_arm_dir: " + result[0]['dir']
            time_test.append([minutes(time.time()),result])

    pickle.dump([time_test,results_dict],open(dir+'/results.pkl','w'))

def sha_finite(model,params,units, n, s, eta, R,dir):
    arms = model.generate_arms(n,dir,params)
    remaining_arms=[list(a) for a in zip(arms.keys(),[0]*len(arms.keys()),[0]*len(arms.keys()),[0]*len(arms.keys()))]
    for i in range(s+1):
        num_pulls = int(R*eta**(i-s))
        num_arms = int( n*eta**(-i))
        print '%d\t%d' %(num_arms,num_pulls)
        for a in range(len(remaining_arms)):
            arm_key=remaining_arms[a][0]
            train_loss,val_acc,test_acc=model.run_solver(units, num_pulls,arms[arm_key])
            print arm_key, train_loss, val_acc, test_acc
            arms[arm_key]['results'].append([num_pulls,train_loss,val_acc,test_acc])
            remaining_arms[a][1]=train_loss
            remaining_arms[a][2]=val_acc
            remaining_arms[a][3]=test_acc
        remaining_arms=sorted(remaining_arms,key=lambda a: -a[2])
        n_k1 = int( n*eta**(-i-1) )
        if s-i-1>=0:
            remaining_arms=remaining_arms[0:n_k1]
    best_arm=arms[remaining_arms[0][0]]
    return arms,[best_arm,remaining_arms[0][1],remaining_arms[0][2],remaining_arms[0][3]]

def main(argv):

    model=''
    data_dir=''
    output_dir=''
    seed_id=0
    device_id=0
    try:
        opts, args = getopt.getopt(argv,"hm:i:o:s:d:",['model=','input_dir=','output_dir=','seed=','device='])
    except getopt.GetoptError:
        print 'hyperband_alg.py -m <model> -i <data_dir> -o <output_dir> -s <rng_seed> -d <GPU_id>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'hyperband_alg.py -i <data_dir> -o <output_dir> -s <rng_seed> -d <GPU_id>'
            sys.exit()
        elif opt in ("-m", "--model"):
            model = arg
        elif opt in ("-i", "--input_dir"):
            data_dir = arg
        elif opt in ("-o", "--output_dir"):
            output_dir = arg
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
        params = get_cnn_search_space()
        obj=cifar10_conv(data_dir,device=device_id,seed=seed_id)
        hyperband_finite(obj,360,'iter',dir,params,100,30000)
    elif model=='svhn':
        params = get_cnn_search_space()
        obj=svhn_conv(data_dir,device=device_id,seed=seed_id)
        hyperband_finite(obj,720,'iter',dir,params,100,60000)
    elif model=='mrbi':
        params = get_cnn_search_space()
        obj=mrbi_conv(data_dir,device=device_id,seed=seed_id)
        hyperband_finite(obj,360,'iter',dir,params,100,30000)
    elif model=='cifar100':
        params = get_nin_search_space()
        obj=nin_conv("cifar100",data_dir,device_id,seed_id)
        hyperband_finite(obj,2000,'iter',dir,params,100,60000)
    elif model=='mnist_svm':
        params= get_svm_search()
        obj=svm_model('mnist_svm',data_dir,seed_id)
        hyperband_finite(obj,24*60,'iter',dir,params,100,40000)


if __name__ == "__main__":
    main(sys.argv[1:])