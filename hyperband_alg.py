import time
import numpy
import pickle
import os
import sys
from cifar10.cifar10_helper import cifar10_conv



class Logger(object):
    def __init__(self,dir):
        self.terminal = sys.stdout
        self.log = open(dir+"/hyperband_run.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()


def sha_inf(model,units,n,B):
    arms = model.generate_arms(n,"/home/lisha/school/Projects/hyperband_nnet/hyperband2/cifar10/hyperband")
    halvings = max(1,int(numpy.ceil(numpy.log2(n))))
    remaining_arms=[list(a) for a in zip(arms.keys(),[0]*len(arms.keys()),[0]*len(arms.keys()),[0]*len(arms.keys()))]
    for i in range(halvings):
        n_arms = int(n/2**i)
        b_arm = B/n_arms/halvings
        print "sha n_arms and budget per arm:" +str(n_arms)+","+ str(b_arm)
        for a in range(len(remaining_arms)):
            arm_key=remaining_arms[a][0]
            train_loss,val_acc, test_acc=model.run_solver(units,b_arm,arms[arm_key])
            arms[arm_key]['results'].append([b_arm,train_loss,val_acc,test_acc])
            remaining_arms[a][1]=train_loss
            remaining_arms[a][2]=val_acc
            remaining_arms[a][3]=test_acc
        remaining_arms=sorted(remaining_arms,key=lambda a: -a[2])[0:max(1,int(numpy.ceil(n_arms/2)))]
    best_arm=arms[remaining_arms[0][0]]
    return arms,[best_arm,remaining_arms[0][1],remaining_arms[0][2],remaining_arms[0][2]]



def hyperband_inf(model,runtime,units,min_unit=10):
    # input t in minutes
    t_0 = time.time()
    print time.localtime(t_0)
    def minutes(t):
        return (t-t_0)/60.
    k=0
    best_acc=0
    results_dict={}
    time_test=[]
    while minutes(time.time())< runtime:
        print "time elapsed: "+ str(minutes(time.time()))
        k+=1
        l=0
        B=2**k
        while k-l >= numpy.log2(l) and minutes(time.time())< runtime:
            n=2**l
            if B/n/max(1,numpy.ceil(numpy.log2(n)))>min_unit:
                arms,result = sha_inf(model,units, n,B)
                results_dict[(k,l)]=arms
                best_acc = max(best_acc,result[2])
                print "k="+str(k)+", l="+str(l)+", best_acc="+str(best_acc)
                time_test.append([minutes(time.time()),best_acc])
            l+=1
    print minutes(time.time())
    print time.localtime(time.time())
    pickle.dump([time_test,results_dict],open('/home/lisha/school/Projects/hyperband_nnet/hyperband2/cifar10/hyperband_2/results.pkl','w'))

def hyperband_finite(model,runtime,units,dir,bounded=True, min_units=100,max_units=60000):
    # input t in minutes
    t_0 = time.time()
    print time.localtime(t_0)
    def minutes(t):
        return (t-t_0)/60.
    k=2
    results_dict={}
    time_test=[]
    while minutes(time.time())< runtime:
        print "time elapsed: "+ str(minutes(time.time()))
        B = int((2**k)*max_units)
        eta = 4.
        def logeta(x):
            return numpy.log(x)/numpy.log(eta)
        #B=(int(logeta(max_units/min_units))+1)*max_units

        k+=1
        if bounded:
            max_halvings = int(numpy.log(max_units/min_units)/numpy.log(eta))
            k = min(numpy.ceil(numpy.log2(max_halvings+1)),k)
        print "\nBudget B = %d" % B
        print '###################'


        # s_max defines the number of inner loops per unique value of B
        # it also specifies the maximum number of rounds
        R = float(max_units)
        r = float(min_units)
        ell_max = int(min(B/R-1,int(logeta(R/r))))
        ell = ell_max

        while ell >= 0 and minutes(time.time())< runtime:

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
                arms,result = sha_finite(model,units, n,s,eta,R,dir)
                results_dict[(k,ell)]=arms
                print "k="+str(k)+", l="+str(ell)+", val_acc="+str(result[2])+", test_acc="+str(result[3])+" best_arm_dir: " + result[0]['dir']
                time_test.append([minutes(time.time()),result])

                ell-=1
        print minutes(time.time())
        print time.localtime(time.time())
        pickle.dump([time_test,results_dict],open(dir+'/results.pkl','w'))

def sha_finite(model,units, n, s, eta, R,dir):
    arms = model.generate_arms(n,dir)
    remaining_arms=[list(a) for a in zip(arms.keys(),[0]*len(arms.keys()),[0]*len(arms.keys()),[0]*len(arms.keys()))]
    for i in range(s+1):
        num_pulls = int(R*eta**(i-s))
        num_arms = int( n*eta**(-i))
        print '%d\t%d' %(num_arms,num_pulls)
        for a in range(len(remaining_arms)):
            arm_key=remaining_arms[a][0]
            train_loss,val_acc,test_acc=model.run_solver(units, num_pulls,arms[arm_key])
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

def main():
    dir='/home/lisha/school/Projects/hyperband_nnet/hyperband2/cifar10/hyperband/trial7'
    #Starting 6 used increasing budget, before used constant budget for max metaarms
    if not os.path.exists(dir):
        os.makedirs(dir)
    sys.stdout = Logger(dir)
    cifar_model=cifar10_conv(device=1,seed=7)
    #hyperband_inf(cifar_model,0.05)
    hyperband_finite(cifar_model,600,'iter',dir)

if __name__ == "__main__":
    main()