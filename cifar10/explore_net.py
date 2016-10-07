import caffe
import pickle
import os
import numpy
import sys
import ast
#Globals
base_lr = 0.001
weight_decay= 0.004
weight_param = dict(lr_mult=1, decay_mult=1)
bias_param   = dict(lr_mult=2, decay_mult=0)
learned_param = [weight_param, bias_param]
def build_net(arm, data_file,mean_file,split='train'):
    def conv_layer(bottom, ks=5, nout=32, stride=1, pad=2, param=learned_param,
                  weight_filler=dict(type='gaussian', std=0.0001),
                  bias_filler=dict(type='constant')):
        conv = caffe.layers.Convolution(bottom, kernel_size=ks, stride=stride,
                             num_output=nout, pad=pad, param=param, weight_filler=weight_filler,
                             bias_filler=bias_filler)
        return conv

    def pooling_layer(bottom, type='ave', ks=3, stride=2):
        if type=='ave':
            return caffe.layers.Pooling(bottom, pool=caffe.params.Pooling.AVE, kernel_size=ks, stride=stride)
        return caffe.layers.Pooling(bottom, pool=caffe.params.Pooling.MAX, kernel_size=ks, stride=stride)

    n = caffe.NetSpec()
    if split=='train':
        n.data, n.label = caffe.layers.Data(batch_size=arm['batch_size'], backend=caffe.params.Data.LMDB, source=data_file,transform_param=dict(mean_file=mean_file),ntop=2)
    else:
        n.data, n.label = caffe.layers.Data(batch_size=arm['batch_size'], backend=caffe.params.Data.LMDB, source="/home/lisha/school/caffe/examples/cifar10/cifar10_test_lmdb",transform_param=dict(mean_file=mean_file),ntop=2)
    n.conv1 = conv_layer(n.data, 5, 32, pad=2, stride=1, param=[dict(lr_mult=1,decay_mult=arm['weight_cost1']/weight_decay),bias_param],weight_filler=dict(type='gaussian', std=arm['init_std1']),
            bias_filler=dict(type='constant'))
    n.pool1 = pooling_layer(n.conv1, 'max', 3, stride=2)
    n.relu1 = caffe.layers.ReLU(n.pool1,in_place=True)
    n.norm1 = caffe.layers.LRN(n.pool1, local_size=3, alpha=arm['scale'], beta=arm['power'], norm_region=1)
    n.conv2 = conv_layer(n.norm1, 5, 32, pad=2, stride=1, param=[dict(lr_mult=1,decay_mult=arm['weight_cost2']/weight_decay),bias_param],weight_filler=dict(type='gaussian', std=arm['init_std2']),
            bias_filler=dict(type='constant'))
    n.relu2 = caffe.layers.ReLU(n.conv2, in_place=True)
    n.pool2 = pooling_layer(n.conv2, 'ave', 3, stride=2)
    n.norm2 = caffe.layers.LRN(n.pool2, local_size=3, alpha=arm['scale'], beta=arm['power'], norm_region=1)
    n.conv3 = conv_layer(n.norm2, 5, 64, pad=2, stride=1, param=[dict(lr_mult=1,decay_mult=arm['weight_cost3']/weight_decay),bias_param],weight_filler=dict(type='gaussian', std=arm['init_std3']),
            bias_filler=dict(type='constant'))
    n.relu3 = caffe.layers.ReLU(n.conv3, in_place=True)
    n.pool3 = pooling_layer(n.conv3, 'ave', 3, stride=2)
    n.ip1 = caffe.layers.InnerProduct(n.pool3, num_output=10, param=[dict(lr_mult=1,decay_mult=arm['weight_cost4']/weight_decay),bias_param],weight_filler=dict(type='gaussian', std=arm['init_std4']),
            bias_filler=dict(type='constant'))
    n.loss = caffe.layers.SoftmaxWithLoss(n.ip1, n.label)
    filename=arm['dir']+'/network_fulltrain.prototxt'
    if split=='test':
        n.acc = caffe.layers.Accuracy(n.ip1, n.label)
        filename=arm['dir']+'/network_test.prototxt'
    with open(filename,'w') as f:
        f.write(str(n.to_proto()))
        return f.name
def build_solver(arm,data_file,mean_file,iters):
    s = caffe.proto.caffe_pb2.SolverParameter()

    #s.random_seed=42
    # Specify locations of the train and (maybe) test networks.
    s.train_net = build_net(arm,data_file,mean_file)
    s.test_net.append(build_net(arm,data_file,mean_file,'test'))
    s.test_interval = 200000  # Test after every 1000 training iterations.
    s.test_iter.append(int(10000/arm['batch_size'])) # Test on 100 batches each time we test.
    #s.test_iter.append(int(10000/arm['batch_size'])) # Test on 100 batches each time we test.

    # The number of iterations over which to average the gradient.
    # Effectively boosts the training batch size by the given factor, without
    # affecting memory utilization.
    s.iter_size = 1

    # 150 epochs max
    s.max_iter = iters     # # of times to update the net (training iterations)

    # Solve using the stochastic gradient descent (SGD) algorithm.
    # Other choices include 'Adam' and 'RMSProp'.
    s.type = 'SGD'

    # Set the initial learning rate for SGD.
    s.base_lr = arm['learning_rate']

    # Set `lr_policy` to define how the learning rate changes during training.
    # Here, we 'step' the learning rate by multiplying it by a factor `gamma`
    # every `stepsize` iterations.
    s.lr_policy = 'step'
    s.gamma = 0.1
    s.stepsize = int(s.max_iter/arm['lr_step'])

    # Set other SGD hyperparameters. Setting a non-zero `momentum` takes a
    # weighted average of the current gradient and previous gradients to make
    # learning more stable. L2 weight decay regularizes learning, to help prevent
    # the model from overfitting.
    s.momentum = 0.9
    s.weight_decay = weight_decay

    # Display the current training loss and accuracy every 1000 iterations.
    s.display = 1000

    # Snapshots are files used to store networks we've trained.  Here, we'll
    # snapshot every 10K iterations -- ten times during training.
    s.random_seed=10
    s.snapshot = 10000
    s.snapshot_prefix = arm['dir']+"/full_cifar10_data"

    # Train on the GPU.  Using the CPU to train large networks is very slow.
    s.solver_mode = caffe.proto.caffe_pb2.SolverParameter.GPU

    # Write the solver to a temporary file and return its filename.
    filename=arm['dir']+"/network_fullsolver.prototxt"
    with open(filename,'w') as f:
        f.write(str(s))
        return f.name
def run_solver(n_units, arm, device,iters):
    #print(arm['dir'])
    caffe.set_device(device)
    caffe.set_mode_gpu()
    dir="/home/lisha/school/caffe/examples/cifar10/"
    s = caffe.get_solver(build_solver(arm,dir+"cifar10_fulltrain_lmdb",dir+"fullmean.binaryproto",iters))

    #prefix=arm['dir']+"/cifar10_data_iter_"
    #s.restore(prefix+str(60000)+".solverstate")
    #s.net.copy_from(prefix+str(60000)+".caffemodel")
    #s.test_nets[0].share_with(s.net)
    #s.test_nets[1].share_with(s.net)
    for i in range(n_units/1000):
        s.step(1000)
        train_loss = s.net.blobs['loss'].data
        test_acc=0
        batches=100
        for i in range(batches):
            s.test_nets[0].forward()
            test_acc += s.test_nets[0].blobs['acc'].data
        test_acc=test_acc/batches
        print train_loss,test_acc
        with open('/home/lisha/school/Projects/hyperband_nnet/hyperband2/cifar10/test_random_seed2.txt','a') as file:
            file.write(arm['dir']+', test_acc, '+str(test_acc)+'\n')
    return test_acc
def calculate_acc(arm, iter):
    caffe.set_device(2)
    caffe.set_mode_gpu()
    dir="/home/lisha/school/caffe/examples/cifar10/"
    s = caffe.get_solver(build_solver(arm,dir+"cifar10_fulltrain_lmdb",dir+"mean.binaryproto"))
    prefix=arm['dir']+"/cifar10_data_iter_"
    s.restore(prefix+str(iter)+".solverstate")
    s.net.copy_from(prefix+str(iter)+".caffemodel")
    s.test_nets[0].share_with(s.net)
    #s.test_nets[1].share_with(s.net)
    train_loss = s.net.blobs['loss'].data
    test_acc=0
    batches=100
    for i in range(batches):
        s.test_nets[0].forward()
        test_acc += s.test_nets[0].blobs['acc'].data
    test_acc=test_acc/batches
    print train_loss,test_acc
def run_hyperband_dir(dir,device,iters):
    os.chdir(dir)
    data=pickle.load(open('results.pkl','r'))
    inds=[0,1,2,3,4,6,7,8,9,10]
    val_acc=[data[0][i][1][2] for i in inds]
    ind_best=val_acc.index(max(val_acc))
    arm=data[0][inds[ind_best]][1][0]
    arm['dir']=os.getcwd()+"/"+arm['dir'][arm['dir'].index('arm'):]
    test_acc=run_solver(iters,arm,device,iters)
    #calculate_acc(arm,100000)
def run_hyperband_inf(dir,device,iters):
    os.chdir(dir)
    data=pickle.load(open('results_inf.pkl','r'))
    best_ind=numpy.argmax([d[1] for d in data[0]])
    keys=data[1].keys()
    keys.sort(key=lambda x:x[0]*10-1*x[1])

    print keys[best_ind]
    data_dict=data[1][keys[best_ind]]
    best_arm=numpy.argmax([float(data_dict[t]['results'][-1][2]) for t in data_dict.keys()])
    arm=data_dict[best_arm]
    arm['dir']=os.getcwd()+"/"+arm['dir'][arm['dir'].index('arm'):]
    test_acc=run_solver(iters,arm,device,iters)
    #calculate_acc(arm,100000)

def run_other_dir(dir,filename):
    os.chdir(dir)
    data=pickle.load(open(filename,'r'))
    val_errors=[t['result'] for t in data['trials']]
    best_arm=numpy.argmin(val_errors[0:56])
    arm_dir=os.getcwd()+'/arm'+str(best_arm+1)
    arm = create_arm_dict(data['trials'][best_arm]['params'],arm_dir)
    run_solver(50*75,arm)
def main():
    device=int(sys.argv[1])
    #rootdir='/home/lisha/school/Projects/hyperband_nnet/hyperband2/cifar10/'
    file_txt=open('explore_params.txt')
    params=file_txt.read()
    params=ast.literal_eval(params)
    arm = create_arm_dict(params,params['dir'])
    run_solver(500*300,arm,device,500*300)
    #run_hyperband_dir(rootdir+'hyperband/trial300',0)
    #if device==0:
    #    run_hyperband_dir(rootdir+'hyperband/trial400',0)
    #    run_hyperband_dir(rootdir+'hyperband/trial500',0)
    #run_hyperband_dir(rootdir+'hyperband/trial1000',1)
    #if device==0:
    #run_hyperband_inf(rootdir+'hyperband_inf/trial22000',device,500*300)
    #run_hyperband_dir(rootdir+'hyperband_constant/unbounded/trial5900',device,500*150)




    #    run_hyperband_dir(rootdir+'hyperband/trial4000',0)
    #else:
    #    run_hyperband_dir(rootdir+'hyperband/trial5000',1)
    #    run_hyperband_dir(rootdir+'hyperband/trial6000',1)
    #    run_hyperband_dir(rootdir+'hyperband/trial7000',1)
    #    run_hyperband_dir(rootdir+'hyperband/trial10000',1)
    #smac_runs=glob.glob(rootdir+"smac/*/")
    #for s in smac_runs:
    #    run_other_dir(s,'smac_2_06_01-dev.pkl')
    #tpe_runs=glob.glob(rootdir+"hyperopt/*/")
    #for s in tpe_runs:
    #    run_other_dir(s,'hyperopt_august2013_mod.pkl')
    #spearmint_runs=glob.glob(rootdir+"spearmint/*/")
    #for s in spearmint_runs:
    #    run_other_dir(s,'spearmint_april2013_mod.pkl')

    #calculate_acc(arm,100000)
def create_arm_dict(trial,arm_dir):
    arm={}
    for k in trial.keys():
        try:
            #arm[k[1:]]=float(trial[k])
            arm[k]=float(trial[k])
        except:
            pass
    arm['lr_step']=int(arm['lr_step'])
    arm['test_net_file']=arm_dir+'/network_test.prototxt'
    arm['batch_size']=100
    arm['n_iter']=0
    arm['dir']=arm_dir
    arm['init_std1']= 0.0001
    arm['init_std2']=0.01
    arm['init_std3']=0.01
    arm['init_std4']=0.01
    return arm
if __name__ == "__main__":
    main()
