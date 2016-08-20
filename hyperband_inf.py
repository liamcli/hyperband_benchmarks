import time
import numpy
import pickle


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



def hyperband_inf(model,runtime,units,min_unit=100):
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
