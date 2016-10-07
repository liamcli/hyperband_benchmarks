import random
import os
import numpy
from params import Param
os.environ['GLOG_minloglevel'] = '1'
import caffe
from model_def import ModelInf
import time
import sys

#Globals
base_lr = 0.001
weight_decay= 0.004
weight_param = dict(lr_mult=1, decay_mult=1)
bias_param   = dict(lr_mult=2, decay_mult=0)
learned_param = [weight_param, bias_param]
#self.data_dir="/home/lisha/school/caffe/examples/cifar10"


class mrbi_conv(ModelInf):
    def __init__(self,data_dir,device=0,seed=1,max_iter=30000):
        self.data_dir=data_dir
        self.name="mrbi_conv"
        caffe.set_device(device)
        caffe.set_mode_gpu()
        self.device=device
        self.max_iter=max_iter
        self.seed=seed
        numpy.random.seed(seed)


    def generate_arms(self,n,dir, params,default=False):
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
                n.data, n.label = caffe.layers.Data(batch_size=arm['batch_size'], backend=caffe.params.Data.LMDB, source=self.data_dir+"/mrbi_train",
                                     transform_param=dict(mean_file=self.data_dir+"/mean.binaryproto"),ntop=2)
                #transform_param=dict(mean_file=self.data_dir+"/mean.binaryproto"),
            elif split==2:
                n.data, n.label = caffe.layers.Data(batch_size=arm['batch_size'], backend=caffe.params.Data.LMDB, source=self.data_dir+"/mrbi_val",
                                     transform_param=dict(mean_file=self.data_dir+"/mean.binaryproto"),ntop=2)
            elif split==3:
                n.data, n.label = caffe.layers.Data(batch_size=arm['batch_size'], backend=caffe.params.Data.LMDB, source=self.data_dir+"/mrbi_test",
                                     transform_param=dict(mean_file=self.data_dir+"/mean.binaryproto"),ntop=2)
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

            #s.random_seed=42
            # Specify locations of the train and (maybe) test networks.
            s.train_net = arm['train_net_file']
            s.test_net.append(arm['val_net_file'])
            s.test_net.append(arm['test_net_file'])
            s.test_interval = 60000  # Test after every 1000 training iterations.
            s.test_iter.append(int(2000/arm['batch_size'])) # Test on 100 batches each time we test.
            s.test_iter.append(int(50000/arm['batch_size'])) # Test on 100 batches each time we test.

            # The number of iterations over which to average the gradient.
            # Effectively boosts the training batch size by the given factor, without
            # affecting memory utilization.
            s.iter_size = 1

            # 150 epochs max
            s.max_iter = self.max_iter     # # of times to update the net (training iterations)

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
            s.stepsize = self.max_iter/arm['lr_step']

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
            s.random_seed=self.seed+int(arm['dir'][arm['dir'].index('arm')+3:])

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
            arm['momentum']=0.90
            arm['learning_rate']=0.003419395
            arm['weight_cost1']=0.063023007
            arm['weight_cost2']=0.0006438862
            arm['weight_cost3']=0.0017623526
            arm['weight_cost4']=0.6515652
            #arm['size']=3
            arm['scale']=0.0001410592
            arm['power']=1.175771
            arm['batch_size']=100
            arm['lr_step']=3
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
            #before 1400
            #arm['learning_rate']=5*10**random.uniform(-5,-1)
            hps=['learning_rate','weight_cost1','weight_cost2','weight_cost3','weight_cost4','scale','power','lr_step']
            for hp in hps:
                val=params[hp].get_param_range(1,stochastic=True)
                arm[hp]=val[0]
            #arm['learning_rate']=5*10**random.uniform(-5,0)
            #arm['weight_cost1']=5*10**random.uniform(-5,0)
            #arm['weight_cost2']=5*10**random.uniform(-5,0)
            #arm['weight_cost3']=5*10**random.uniform(-5,0)
            #arm['weight_cost4']=5*10**random.uniform(-3,2)
            #arm['size']=3
            #arm['scale']=5*10**random.uniform(-6,0)
            #before 2000
            #arm['power']=random.uniform(0.25,5)
            #int(10**random.uniform(2,4)/100)*100
            #arm['power']=random.uniform(0.01,3)
            arm['batch_size']=100
            #arm['lr_step']=int(random.uniform(1,6))*10000
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
            s.test_nets[0].share_with(s.net)
            s.test_nets[1].share_with(s.net)
        start=time.time()
        if unit=='time':
            while time.time()-start<n_units:
                s.step(1)
                arm['n_iter']+=1
                #print time.localtime(time.time())
        elif unit=='iter':
            n_units=min(n_units,self.max_iter-arm['n_iter'])
            s.step(n_units)
            arm['n_iter']+=n_units
        s.snapshot()
        train_loss = s.net.blobs['loss'].data
        val_acc=0
        test_acc=0
        test_batches=500
        val_batches=20
        for i in range(val_batches):
            s.test_nets[0].forward()
            val_acc += s.test_nets[0].blobs['acc'].data
        if arm['n_iter']==self.max_iter:
            for i in range(test_batches):
                s.test_nets[1].forward()
                test_acc += s.test_nets[1].blobs['acc'].data

        val_acc=val_acc/val_batches
        test_acc=test_acc/test_batches
        return train_loss,val_acc, test_acc
def get_cnn_search_space():
    params={}
    params['learning_rate']=Param('learning_rate',numpy.log(5*10**(-5)),numpy.log(5),distrib='uniform',scale='log')
    params['weight_cost1']=Param('weight_cost1',numpy.log(5*10**(-5)),numpy.log(5),distrib='uniform',scale='log')
    params['weight_cost2']=Param('weight_cost2',numpy.log(5*10**(-5)),numpy.log(5),distrib='uniform',scale='log')
    params['weight_cost3']=Param('weight_cost3',numpy.log(5*10**(-5)),numpy.log(5),distrib='uniform',scale='log')
    params['weight_cost4']=Param('weight_cost4',numpy.log(5*10**(-3)),numpy.log(500),distrib='uniform',scale='log')
    #params['momentum']=Param('momentum',0,1,distrib='uniform',scale='linear')
    #arm['size']=3
    params['scale']=Param('scale',numpy.log(5*10**(-6)),numpy.log(5),distrib='uniform',scale='log')
    #before 2000
    #arm['power']=random.uniform(0.25,5)
    #int(10**random.uniform(2,4)/100)*100
    params['power']=Param('power',0.01,3,distrib='uniform',scale='linear')
    params['lr_step']=Param('lr_step',1,5,distrib='uniform',scale='linear',interval=1)
    return params

def main():
    #data_dir=sys.argv[1]
    #output_dir=sys.argv[2]
    data_dir="/home/lisha/school/Projects/hyperband_nnet/hyperband2/mrbi"
    output_dir="/home/lisha/school/Projects/hyperband_nnet/hyperband2/mrbi/default"
    model= mrbi_conv(data_dir,device=1)
    param=get_cnn_search_space()
    arms = model.generate_arms(1,output_dir,param,True)
    train_loss,val_acc, test_acc = model.run_solver('iter',30000,arms[0])
    print train_loss, val_acc, test_acc



if __name__ == "__main__":
    main()
