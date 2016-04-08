import time
import os
import pickle
from cifar10.cifar10_helper import cifar10_conv

def random(model,units,runtime,dir,max_units=60000):
    arms=[]
    start_time=time.time()
    results=[]
    def minutes(t):
        return (t-start_time)/60.
    while minutes(time.time())<runtime:
        arm = model.generate_arms(1,dir)[0]
        train_loss,val_acc, test_acc=model.run_solver(units,max_units,arm)
        arm['results'].append([train_loss,val_acc,test_acc])
        arms.append(arm)
        duration=minutes(time.time())
        results.append([duration,train_loss, val_acc,test_acc])
        print "duration: "+str(duration) + " train_err: "+str(train_loss) + " val_acc: " + str(val_acc) + " test_acc: " + str(test_acc)
    pickle.dump([results,arms],open(dir+'/results.pkl','w'))

def main():
    cifar_model=cifar10_conv(device=0,seed=2)
    dir="/home/lisha/school/Projects/hyperband_nnet/hyperband2/cifar10/random/trial2"
    if not os.path.exists(dir):
        os.makedirs(dir)
    #hyperband_inf(cifar_model,0.05)
    random(cifar_model,'iter',600,dir)

if __name__ == "__main__":
    main()