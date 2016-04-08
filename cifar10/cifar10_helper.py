import random
import os
os.environ['GLOG_minloglevel'] = '1'
import caffe
from model_def import ModelInf
import time

#Globals
base_lr = 0.001
weight_decay= 0.004
weight_param = dict(lr_mult=1, decay_mult=1)
bias_param   = dict(lr_mult=2, decay_mult=0)
learned_param = [weight_param, bias_param]
data_dir="/home/lisha/school/caffe/examples/cifar10"


class cifar10_conv(ModelInf):
    def __init__(self,device=0,seed=1):
        self.name="cifar10_conv"
        caffe.set_device(device)
        caffe.set_mode_gpu()
        self.device=device
        random.seed(seed)


    def generate_arms(self,n,dir, default=False):
        def build_net(arm, split=0):
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
            if split==1:
                n.data, n.label = caffe.layers.Data(batch_size=arm['batch_size'], backend=caffe.params.Data.LMDB, source=data_dir+"/cifar10_train_lmdb",
                                     transform_param=dict(mean_file=data_dir+"/mean.binaryproto"),ntop=2)
                #transform_param=dict(mean_file=data_dir+"/mean.binaryproto"),
            elif split==2:
                n.data, n.label = caffe.layers.Data(batch_size=arm['batch_size'], backend=caffe.params.Data.LMDB, source=data_dir+"/cifar10_val_lmdb",
                                     transform_param=dict(mean_file=data_dir+"/mean.binaryproto"),ntop=2)
            elif split==3:
                n.data, n.label = caffe.layers.Data(batch_size=arm['batch_size'], backend=caffe.params.Data.LMDB, source=data_dir+"/cifar10_test_lmdb",
                                     transform_param=dict(mean_file=data_dir+"/mean.binaryproto"),ntop=2)
            n.conv1 = conv_layer(n.data, 5, 32, pad=2, stride=1, param=[dict(lr_mult=1,decay_mult=arm['weight_cost1']/weight_decay),bias_param],weight_filler=dict(type='gaussian', std=arm['init_std1']),
                    bias_filler=dict(type='constant'))
            n.pool1 = pooling_layer(n.conv1, 'max', 3, stride=2)
            n.relu1 = caffe.layers.ReLU(n.pool1,in_place=True)
            n.norm1 = caffe.layers.LRN(n.pool1, local_size=arm['size'], alpha=arm['scale'], beta=arm['power'], norm_region=1)
            n.conv2 = conv_layer(n.norm1, 5, 32, pad=2, stride=1, param=[dict(lr_mult=1,decay_mult=arm['weight_cost2']/weight_decay),bias_param],weight_filler=dict(type='gaussian', std=arm['init_std2']),
                    bias_filler=dict(type='constant'))
            n.relu2 = caffe.layers.ReLU(n.conv2, in_place=True)
            n.pool2 = pooling_layer(n.conv2, 'ave', 3, stride=2)
            n.norm2 = caffe.layers.LRN(n.pool2, local_size=arm['size'], alpha=arm['scale'], beta=arm['power'], norm_region=1)
            n.conv3 = conv_layer(n.norm2, 5, 64, pad=2, stride=1, param=[dict(lr_mult=1,decay_mult=arm['weight_cost3']/weight_decay),bias_param],weight_filler=dict(type='gaussian', std=arm['init_std3']),
                    bias_filler=dict(type='constant'))
            n.relu3 = caffe.layers.ReLU(n.conv3, in_place=True)
            n.pool3 = pooling_layer(n.conv3, 'ave', 3, stride=2)
            n.ip1 = caffe.layers.InnerProduct(n.pool3, num_output=10, param=[dict(lr_mult=1,decay_mult=arm['weight_cost4']/weight_decay),bias_param],weight_filler=dict(type='gaussian', std=arm['init_std4']),
                    bias_filler=dict(type='constant'))
            n.loss = caffe.layers.SoftmaxWithLoss(n.ip1, n.label)
            if split==1:
                filename=arm['dir']+'/network_train.prototxt'
            elif split==2:
                n.acc = caffe.layers.Accuracy(n.ip1, n.label)
                filename=arm['dir']+'/network_val.prototxt'
            elif split==3:
                n.acc = caffe.layers.Accuracy(n.ip1, n.label)
                filename=arm['dir']+'/network_test.prototxt'
            with open(filename,'w') as f:
                f.write(str(n.to_proto()))
                return f.name

        def build_solver(arm):
            s = caffe.proto.caffe_pb2.SolverParameter()

            # Specify locations of the train and (maybe) test networks.
            s.train_net = arm['train_net_file']
            s.test_net.append(arm['val_net_file'])
            s.test_net.append(arm['test_net_file'])
            s.test_interval = 1000  # Test after every 1000 training iterations.
            s.test_iter.append(int(10000/arm['batch_size'])) # Test on 100 batches each time we test.
            s.test_iter.append(int(10000/arm['batch_size'])) # Test on 100 batches each time we test.

            # The number of iterations over which to average the gradient.
            # Effectively boosts the training batch size by the given factor, without
            # affecting memory utilization.
            s.iter_size = 1

            # 150 epochs max
            s.max_iter = 40000/arm['batch_size']*150     # # of times to update the net (training iterations)

            # Solve using the stochastic gradient descent (SGD) algorithm.
            # Other choices include 'Adam' and 'RMSProp'.
            s.type = 'SGD'

            # Set the initial learning rate for SGD.
            s.base_lr = arm['learning_rate']

            # Set `lr_policy` to define how the learning rate changes during training.
            # Here, we 'step' the learning rate by multiplying it by a factor `gamma`
            # every `stepsize` iterations.
            s.lr_policy = 'step'
            s.gamma = arm['lr_step']
            s.stepsize = 1

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
            s.snapshot = 10000
            s.snapshot_prefix = arm['dir']+"/cifar10_data"

            # Train on the GPU.  Using the CPU to train large networks is very slow.
            s.solver_mode = caffe.proto.caffe_pb2.SolverParameter.GPU

            # Write the solver to a temporary file and return its filename.
            filename=arm['dir']+"/network_solver.prototxt"
            with open(filename,'w') as f:
                f.write(str(s))
                return f.name

        os.chdir(dir)
        arms={}
        if default:
            arm={}
            dirname="default_arm"
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            arm['dir']=dir+"/"+dirname
            arm['n_iter']=0
            arm['learning_rate']=0.001
            arm['weight_cost1']=0.004
            arm['weight_cost2']=0.004
            arm['weight_cost3']=0.004
            arm['weight_cost4']=1
            arm['size']=3
            arm['scale']=0.00005
            arm['power']=0.75
            arm['batch_size']=100
            arm['lr_step']=1
            arm['init_std1']=0.0001
            arm['init_std2']=0.01
            arm['init_std3']=0.01
            arm['init_std4']=0.01
            arm['train_net_file'] = build_net(arm,1)
            arm['val_net_file'] = build_net(arm,2)
            arm['test_net_file'] = build_net(arm,3)
            arm['solver_file'] = build_solver(arm)
            arm['results']=[]
            arms[0]=arm
            return arms
        subdirs=next(os.walk('.'))[1]
        if len(subdirs)==0:
            start_count=0
        else:
            start_count=len(subdirs)
        for i in range(n):
            dirname="arm"+str(start_count+i)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            arm={}
            arm['dir']=dir+"/"+dirname
            arm['n_iter']=0
            arm['learning_rate']=5*10**random.uniform(-5,-1)
            arm['weight_cost1']=10**random.uniform(-5,0)
            arm['weight_cost2']=10**random.uniform(-5,0)
            arm['weight_cost3']=10**random.uniform(-5,0)
            arm['weight_cost4']=10**random.uniform(-5,0)
            arm['size']=3
            arm['scale']=5*10**random.uniform(-6,0)
            arm['power']=random.uniform(0.25,5)
            #int(10**random.uniform(2,4)/100)*100
            arm['batch_size']=100
            arm['lr_step']=random.uniform(0.995,1)
            arm['init_std1']=0.0001
            arm['init_std2']=0.01
            arm['init_std3']=0.01
            arm['init_std4']=0.01
            #arm['init_std1']=10**random.uniform(-6,-1)
            #arm['init_std2']=10**random.uniform(-6,-1)
            #arm['init_std3']=10**random.uniform(-6,-1)
            #arm['init_std4']=10**random.uniform(-6,-1)
            arm['train_net_file'] = build_net(arm,1)
            arm['val_net_file'] = build_net(arm,2)
            arm['test_net_file'] = build_net(arm,3)
            arm['solver_file'] = build_solver(arm)
            arm['results']=[]
            arms[i]=arm

        return arms

    def run_solver(self, unit, n_units, arm, disp_interval=100):
        #print(arm['dir'])
        caffe.set_device(self.device)
        caffe.set_mode_gpu()
        s = caffe.get_solver(arm['solver_file'])

        if arm['n_iter']>0:
            prefix=arm['dir']+"/cifar10_data_iter_"
            s.restore(prefix+str(arm['n_iter'])+".solverstate")
            s.net.copy_from(prefix+str(arm['n_iter'])+".caffemodel")
        start=time.time()
        if unit=='time':
            while time.time()-start<n_units:
                s.step(1)
                arm['n_iter']+=1
                #print time.localtime(time.time())
        elif unit=='iter':
            n_units=min(n_units,60000-arm['n_iter'])
            s.step(n_units)
            arm['n_iter']+=n_units
        s.snapshot()
        train_loss = s.net.blobs['loss'].data
        s.test_nets[0].forward()
        val_acc = s.test_nets[0].blobs['acc'].data
        s.test_nets[1].forward()
        test_acc = s.test_nets[1].blobs['acc'].data
        return train_loss,val_acc, test_acc

def main():

    model= cifar10_conv()
    arms = model.generate_arms(1,"/home/lisha/school/Projects/hyperband_nnet/hyperband2/cifar10/hyperband/trial2",True)
    train_loss,val_acc, test_acc = model.run_solver('iter',40000,arms[0])
    print train_loss, val_acc, test_acc


if __name__ == "__main__":
    main()