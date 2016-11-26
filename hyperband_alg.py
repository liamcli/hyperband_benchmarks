import time
import numpy
import pickle
import os
import sys,getopt
from params import zoom_space



class Logger(object):
    def __init__(self,dir):
        self.terminal = sys.stdout
        self.log = open(dir+"/hyperband_run.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
def hyperband_finite(model,runtime,units,dir,params,min_units,max_units,eta=4.,B=0,max_k=2,s_run=None,bounded=True,adaptive=False):
    # inputs:
    # model - object with necessary subroutines to generate arms and train models
    # runtime - total time to run the optimization routine
    # units - type of resource that will be allocated choices are "iter" or "time."  iter should be used for everything if
    #         time is not the desired resource
    # dir - output directory to store the files for this run
    # params - object with specified hyperparameter search space from which arms are sampled
    # min_units - minimum units to train any configuration on
    # max_units - maximum units per configuration
    # NOTE: R in the algorithm corresponds to max_units/min_units
    # eta - elimination rate
    # B - budget per bracket of successive halving
    # max_k - # of times to run hyperband, i.e. # of times to repeat the outer loops over the tradeoffs s
    # s_run - option to repeat a specific bracket (takes integer argument, needs to be within possible range of s)
    # bounded - option to toggle whether to use doubling budget in outer loop or constant budget.  True uses constant budget
    # False uses doubling budget.
    t_0 = time.time()
    print time.localtime(t_0)
    def minutes(t):
        return (t-t_0)/60.
    k=0
    results_dict={}
    time_test=[]
    while minutes(time.time())< runtime and k <max_k:

        def logeta(x):
            return numpy.round(numpy.log(x)/numpy.log(eta),decimals=10)
        if B==0:
            if bounded:
                B=int(numpy.floor(logeta(max_units/min_units))+1)*max_units
            else:
                B = int((2**k)*max_units)

        k+=1

        print "\nBudget B = %d" % B
        print '###################'


        # s_max defines the number of inner loops per unique value of B
        # it also specifies the maximum number of rounds
        R = float(max_units)
        r = float(min_units)
        ell_max = int(min(B/R-1,int(numpy.floor(logeta(R/r)))))
        ell = ell_max
        best_val =0
        print ell_max

        while ell >= 0 and minutes(time.time())< runtime:
        #while minutes(time.time())< runtime:

            # specify the number of arms and the number of times each arm is pulled per stage within this innerloop
            n = int( B/R*eta**ell/(ell+1.) )

            if n> 0:
                s = 0
                while (n)*R*(s+1.)*eta**(-s)>B:
                    s+=1
                if s_run is None or s==s_run:
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
                pickle.dump([time_test,results_dict],open(dir+'/results.pkl','w'))
                ell-=1
        # This option is included here for reproducibility only.  We do not recommend using this naive approach to adaptive configuration
        # selection.  REMEMBER!  Exclude brackets that use this adaptive heuristic when aggregating results.
        if adaptive:
            zoom_params=zoom_space(params,best_arm)
            arms,result = sha_finite(model,zoom_params,units, best_n,best_s,eta,R,dir)
            results_dict[(k,ell)]=arms
            print "k="+str(k)+", l="+str(ell)+", val_acc="+str(result[2])+", test_acc="+str(result[3])+" best_arm_dir: " + result[0]['dir']
            time_test.append([minutes(time.time()),result])




def sha_finite(model,params,units, n, s, eta, R,dir):
    arms = model.generate_arms(n,dir,params)
    remaining_arms=[list(a) for a in zip(arms.keys(),[0]*len(arms.keys()),[0]*len(arms.keys()),[0]*len(arms.keys()))]
    for i in range(s+1):
        num_pulls = int(R*eta**(i-s))
        num_arms = int( n*eta**(-i))
        print '%d\t%d' %(num_arms,num_pulls)
        for a in range(len(remaining_arms)):
            start_time=time.time()
            arm_key=remaining_arms[a][0]
            print arms[arm_key]
            train_loss,val_acc,test_acc=model.run_solver(units, num_pulls,arms[arm_key])
            print arm_key, train_loss, val_acc, test_acc, (time.time()-start_time)/60.0
            arms[arm_key]['results'].append([num_pulls,train_loss,val_acc,test_acc])
            remaining_arms[a][1]=train_loss
            remaining_arms[a][2]=val_acc
            remaining_arms[a][3]=test_acc
        remaining_arms=sorted(remaining_arms,key=lambda a: -a[2])
        n_k1 = int( n*eta**(-i-1) )
        if s-i-1>=0:
            for k in range(n_k1,len(remaining_arms)):
                arm_dir=arms[remaining_arms[k][0]]['dir']
                files = os.listdir(arm_dir)
                for file in files:
                    if file.endswith(".caffemodel") or file.endswith(".solverstate"):
                        os.remove(os.path.join(arm_dir,file))
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

    if not os.path.exists(dir):
        os.makedirs(dir)
    sys.stdout = Logger(dir)
    if model=='cifar10':
        from cifar10.cifar10_helper import get_cnn_search_space,cifar10_conv
        params = get_cnn_search_space()
        obj=cifar10_conv(data_dir,device=device_id,seed=seed_id)
        hyperband_finite(obj,360,'iter',dir,params,100,30000,adaptive=True)
    elif model=='cifar10_s4':
        from cifar10.cifar10_helper import get_cnn_search_space,cifar10_conv
        params = get_cnn_search_space()
        obj=cifar10_conv(data_dir,device=device_id,seed=seed_id)
        hyperband_finite(obj,360,'iter',dir,params,100,30000,max_k=10,s_run=4,adaptive=False)
    elif model=='svhn':
        from svhn.svhn_helper import get_cnn_search_space,svhn_conv
        params = get_cnn_search_space()
        obj=svhn_conv(data_dir,device=device_id,seed=seed_id)
        hyperband_finite(obj,720,'iter',dir,params,100,60000,adaptive=True)
    elif model=='svhn_s4':
        from svhn.svhn_helper import get_cnn_search_space,svhn_conv
        params = get_cnn_search_space()
        obj=svhn_conv(data_dir,device=device_id,seed=seed_id)
        hyperband_finite(obj,720,'iter',dir,params,100,60000,max_k=10,s_run=4,adaptive=False)
    elif model=='mrbi':
        from mrbi.mrbi_helper import get_cnn_search_space,mrbi_conv
        params = get_cnn_search_space()
        obj=mrbi_conv(data_dir,device=device_id,seed=seed_id)
        hyperband_finite(obj,360,'iter',dir,params,100,30000,adaptive=True)
    elif model=='mrbi_s4':
        from mrbi.mrbi_helper import get_cnn_search_space,mrbi_conv
        params = get_cnn_search_space()
        obj=mrbi_conv(data_dir,device=device_id,seed=seed_id)
        hyperband_finite(obj,360,'iter',dir,params,100,30000,max_k=10,s_run=4,adaptive=False)
    elif model=='cifar10_svm':
        from kernel.kernel_lsqr_helper import get_svm_search,svm_model
        params= get_svm_search()
        obj=svm_model('cifar10',data_dir,seed_id)
        hyperband_finite(obj,12*60,'iter',dir,params,100,len(obj.orig_data['y_train']), adaptive=False)
    elif model=='cifar10_svm_s4':
        from kernel.kernel_lsqr_helper import get_svm_search,svm_model
        params= get_svm_search()
        obj=svm_model('cifar10',data_dir,seed_id)
        hyperband_finite(obj,12*60,'iter',dir,params,100,len(obj.orig_data['y_train']),max_k=10,s_run=4,adaptive=False)
    elif model=='cifar10_random_features':
        from kernel.random_features_helper import get_svm_search,random_features_model
        params=get_svm_search()
        obj=random_features_model('cifar10',data_dir,seed=seed_id)
        hyperband_finite(obj,12*60,'iter',dir,params,200,100000, adaptive=False)
    elif model=='cifar10_random_features_s4':
        from kernel.random_features_helper import get_svm_search,random_features_model
        params=get_svm_search()
        obj=random_features_model('cifar10',data_dir,seed=seed_id)
        hyperband_finite(obj,12*60,'iter',dir,params,200,100000,max_k=10,s_run=4,adaptive=False)
if __name__ == "__main__":
    main(sys.argv[1:])
