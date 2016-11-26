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
base_lr = 1.0
weight_decay= 1.0
weight_param = dict(lr_mult=1, decay_mult=1)
bias_param   = dict(lr_mult=2, decay_mult=0)
learned_param = [weight_param, bias_param]


class nin_conv(ModelInf):
    def __init__(self,problem,data_dir,device=0,seed=1):
        self.data_dir=data_dir
        self.name="nin_conv"
        caffe.set_device(device)
        caffe.set_mode_gpu()
        self.device=device
        self.seed=seed
        numpy.random.seed(seed)
        self.problem=problem


    def generate_arms(self,n,dir, params,default=False):
        def build_net(arm,split):
            if self.problem=='cifar10':
                n_class = 10
                self.data_dirs=[self.data_dir+'/cifar-zca-ptrain', self.data_dir+'/cifar-zca-pval',self.data_dir+'/cifar-zca-test']
                backend = caffe.params.Data.LMDB
            else:
                n_class = 100
                self.data_dirs=[self.data_dir+'/cifar100_ptrain_lmdb', self.data_dir+'/cifar100_pval_lmdb',self.data_dir+'/cifar100_test_lmdb']
                backend = caffe.params.Data.LMDB
            def conv_layer(bottom, ks=5, nout=32, stride=1, pad=2, group=0,param=learned_param,
                          weight_filler=dict(type='gaussian', std=0.0001),
                          bias_filler=dict(type='constant')):
                if group==0:
                    conv = caffe.layers.Convolution(bottom, kernel_size=ks, stride=stride,
                                     num_output=nout, pad=pad, param=param, weight_filler=weight_filler,
                                     bias_filler=bias_filler)
                else:
                    conv = caffe.layers.Convolution(bottom, kernel_size=ks, stride=stride,
                                     num_output=nout, pad=pad, group=group, param=param, weight_filler=weight_filler,
                                     bias_filler=bias_filler)
                return conv

            def pooling_layer(bottom, type='ave', ks=3, stride=2):
                if type=='ave':
                    return caffe.layers.Pooling(bottom, pool=caffe.params.Pooling.AVE, kernel_size=ks, stride=stride,engine=1)
                return caffe.layers.Pooling(bottom, pool=caffe.params.Pooling.MAX, kernel_size=ks, stride=stride,engine=1)

            n = caffe.NetSpec()
            if split==1:
                n.data, n.label = caffe.layers.Data(batch_size=128, backend=backend, source=self.data_dirs[0],ntop=2)
            elif split==2:
                n.data, n.label = caffe.layers.Data(batch_size=100, backend=backend, source=self.data_dirs[1],ntop=2)
            elif split==3:
                n.data, n.label = caffe.layers.Data(batch_size=100, backend=backend, source=self.data_dirs[2],ntop=2)
            pool_map={1:'ave',2:'max'}
            param1=[dict(lr_mult=arm['learning_rate1'],decay_mult=arm['weight_cost1']),dict(lr_mult=2*arm['learning_rate1'],decay_mult=2*arm['weight_cost1'])]
            param2=[dict(lr_mult=arm['learning_rate2'],decay_mult=arm['weight_cost2']),dict(lr_mult=2*arm['learning_rate2'],decay_mult=2*arm['weight_cost2'])]
            param3=[dict(lr_mult=arm['learning_rate3'],decay_mult=arm['weight_cost3']),dict(lr_mult=2*arm['learning_rate3'],decay_mult=2*arm['weight_cost3'])]
            n.conv1a = conv_layer(n.data, ks=5,nout=192,stride=1,pad=2,group=0,param=param1,
                weight_filler=dict(type='gaussian', std=arm['w_init1']), bias_filler=dict(type='constant'))
            n.relu1a = caffe.layers.ReLU(n.conv1a,in_place=True)
            n.conv1b = conv_layer(n.conv1a, ks=1,nout=160,stride=1,pad=0,group=1,param=param1,
                weight_filler=dict(type='gaussian', std=arm['w_init1']), bias_filler=dict(type='constant'))
            n.relu1b = caffe.layers.ReLU(n.conv1b,in_place=True)
            n.conv1c = conv_layer(n.conv1b, ks=1,nout=96,stride=1,pad=0,group=1,param=param1,
                weight_filler=dict(type='gaussian', std=arm['w_init1']), bias_filler=dict(type='constant'))
            n.relu1c = caffe.layers.ReLU(n.conv1c, in_place=True)
            n.pool1 = pooling_layer(n.conv1c,type=pool_map[2],ks=3,stride=2)
            n.dropout1 = caffe.layers.Dropout(n.pool1,dropout_ratio=arm['dropout1'],in_place=True)
            n.conv2a = conv_layer(n.pool1, ks=5,nout=192,stride=1,pad=2,group=0,param=param2,
                weight_filler=dict(type='gaussian', std=arm['w_init2']), bias_filler=dict(type='constant'))
            n.relu2a = caffe.layers.ReLU(n.conv2a,in_place=True)
            n.conv2b = conv_layer(n.conv2a, ks=1,nout=192,stride=1,pad=0,group=1,param=param2,
                weight_filler=dict(type='gaussian', std=arm['w_init2']), bias_filler=dict(type='constant'))
            n.relu2b = caffe.layers.ReLU(n.conv2b,in_place=True)
            n.conv2c = conv_layer(n.conv2b, ks=1,nout=192,stride=1,pad=0,group=1,param=param2,
                weight_filler=dict(type='gaussian', std=arm['w_init2']), bias_filler=dict(type='constant'))
            n.relu2c = caffe.layers.ReLU(n.conv2c, in_place=True)
            n.pool2 = pooling_layer(n.conv2c,type=pool_map[1],ks=3,stride=2)
            n.dropout2 = caffe.layers.Dropout(n.pool2,dropout_ratio=arm['dropout2'],in_place=True)
            n.conv3a = conv_layer(n.pool2, ks=3,nout=192,stride=1,pad=1,group=0,param=param3,
                weight_filler=dict(type='gaussian', std=arm['w_init3']), bias_filler=dict(type='constant'))
            n.relu3a = caffe.layers.ReLU(n.conv3a,in_place=True)
            n.conv3b = conv_layer(n.conv3a, ks=1,nout=192,stride=1,pad=0,group=1,param=param3,
                weight_filler=dict(type='gaussian', std=arm['w_init3']), bias_filler=dict(type='constant'))
            n.relu3b = caffe.layers.ReLU(n.conv3b,in_place=True)
            n.conv3c = conv_layer(n.conv3b, ks=1,nout=n_class,stride=1,pad=0,group=1,param=param3,
                weight_filler=dict(type='gaussian', std=arm['w_init3']), bias_filler=dict(type='constant'))
            n.relu3c = caffe.layers.ReLU(n.conv3c, in_place=True)
            n.pool3 = pooling_layer(n.conv3c,type=pool_map[1],ks=8,stride=1)
            n.loss = caffe.layers.SoftmaxWithLoss(n.pool3,n.label)

            
            if split==1:
                filename=arm['dir']+'/network_train.prototxt'
            elif split==2:
                n.acc = caffe.layers.Accuracy(n.pool3, n.label)
                filename=arm['dir']+'/network_val.prototxt'
            elif split==3:
                n.acc = caffe.layers.Accuracy(n.pool3, n.label)
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
            s.test_interval = 60000  # Test after every 1000 training iterations.
            s.test_iter.append(int(10000/arm['batch_size'])) # Test on 100 batches each time we test.
            s.test_iter.append(int(10000/arm['batch_size'])) # Test on 100 batches each time we test.

            # The number of iterations over which to average the gradient.
            # Effectively boosts the training batch size by the given factor, without
            # affecting memory utilization.
            s.iter_size = 1

            # 150 epochs max
            s.max_iter = 60000     # # of times to update the net (training iterations)

            # Solve using the stochastic gradient descent (SGD) algorithm.
            # Other choices include 'Adam' and 'RMSProp'.
            #s.type = 'SGD'

            # Set the initial learning rate for SGD.
            s.base_lr = base_lr

            # Set `lr_policy` to define how the learning rate changes during training.
            # Here, we 'step' the learning rate by multiplying it by a factor `gamma`
            # every `stepsize` iterations.
            s.lr_policy = 'multistep'
            s.gamma = 0.1
            #s.stepsize=s.max_iter/arm['lr_step']
            s.stepvalue.append(40000)
            s.stepvalue.append(50000)
            s.stepvalue.append(60000)
            #size=int((s.max_iter-40000)/arm['lr_step'])
            #for j in range(arm['lr_step']):
            #    s.stepvalue.append(40000+(j+1)*size)

            # Set other SGD hyperparameters. Setting a non-zero `momentum` takes a
            # weighted average of the current gradient and previous gradients to make
            # learning more stable. L2 weight decay regularizes learning, to help prevent
            # the model from overfitting.
            s.momentum = arm['momentum']
            s.weight_decay = weight_decay
            s.random_seed=self.seed+int(arm['dir'][arm['dir'].index('arm')+3:])
            # Display the current training loss and accuracy every 1000 iterations.
            s.display = 100

            # Snapshots are files used to store networks we've trained.  Here, we'll
            # snapshot every 10K iterations -- ten times during training.
            s.snapshot = 10000
            s.snapshot_prefix = arm['dir']+"/"+self.problem+"_data"

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
            arm['batch_size']=100
            arm['momentum']=0.9
            arm['learning_rate1']=0.1
            arm['learning_rate2']=0.1
            arm['learning_rate3']=0.01
            arm['weight_cost1']=0.0001
            arm['weight_cost2']=0.0001
            arm['weight_cost3']=0.0001
            arm['dropout1']=0.5
            arm['dropout2']=0.5
            arm['w_init1']=0.05
            arm['w_init2']=0.05
            arm['w_init3']=0.05
            arm['lr_step']=2
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
            hps=['momentum','learning_rate1','learning_rate2','learning_rate3',
            'weight_cost1','weight_cost2','weight_cost3',
            'dropout1','dropout2','w_init1','w_init2','w_init3']
            for hp in hps:
                val=params[hp].get_param_range(1,stochastic=True)
                arm[hp]=val[0]
            
            arm['batch_size']=100
            
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
            prefix=arm['dir']+"/"+str(self.problem)+"_data_iter_"
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
            n_units=min(n_units,400*150-arm['n_iter'])
            s.step(n_units)
            arm['n_iter']+=n_units
        s.snapshot()
        train_loss = s.net.blobs['loss'].data
        val_acc=0
        test_acc=0
        test_batches=100
        val_batches=100
        for i in range(val_batches):
            s.test_nets[0].forward()
            val_acc += s.test_nets[0].blobs['acc'].data
        for i in range(test_batches):
            s.test_nets[1].forward()
            test_acc += s.test_nets[1].blobs['acc'].data

        val_acc=val_acc/val_batches
        test_acc=test_acc/test_batches
        del s
        return train_loss,val_acc, test_acc
def get_nin_search_space():
    params={}
    params['momentum']=Param('momentum',0.8,1,distrib='uniform',scale='linear')
    params['learning_rate1']=Param('learning_rate1',numpy.log(5*10**(-3)),numpy.log(0.5),distrib='uniform',scale='log')
    params['learning_rate2']=Param('learning_rate2',numpy.log(5*10**(-3)),numpy.log(0.5),distrib='uniform',scale='log')
    params['learning_rate3']=Param('learning_rate3',numpy.log(5*10**(-3)),numpy.log(0.5),distrib='uniform',scale='log')
    params['weight_cost1']=Param('weight_cost1',numpy.log(10**(-5)),numpy.log(10**(-3)),distrib='uniform',scale='log')
    params['weight_cost2']=Param('weight_cost2',numpy.log(10**(-5)),numpy.log(10**(-3)),distrib='uniform',scale='log')
    params['weight_cost3']=Param('weight_cost3',numpy.log(10**(-5)),numpy.log(10**(-3)),distrib='uniform',scale='log')
    params['dropout1']=Param('dropout1',0.4,0.6,distrib='uniform',scale='linear')
    params['dropout2']=Param('dropout2',0.4,0.6,distrib='uniform',scale='linear')
    params['w_init1']=Param('w_init1',numpy.log(10**(-2)),0.0,distrib='uniform',scale='log')
    params['w_init2']=Param('w_init2',numpy.log(10**(-2)),0.0,distrib='uniform',scale='log')
    params['w_init3']=Param('w_init3',numpy.log(10**(-2)),0.0,distrib='uniform',scale='log')
    #params['pool1']=Param('pool1',1,3,distrib='uniform',scale='linear',interval=1)
    #params['pool2']=Param('pool2',1,3,distrib='uniform',scale='linear',interval=1)
    #params['pool3']=Param('pool3',1,3,distrib='uniform',scale='linear',interval=1)
    return params

def main():
    data_dir=sys.argv[1]
    output_dir=sys.argv[2]
    #"/home/lisha/school/caffe/examples/cifar10"
    model= nin_conv("cifar100",data_dir,device=1)
    param=get_nin_search_space()
    #"/home/lisha/school/Projects/hyperband_nnet/hyperband2/cifar10/default"
    arms = model.generate_arms(1,output_dir,param,True)
    train_loss,val_acc, test_acc = model.run_solver('iter',5000,arms[0])
    print train_loss, val_acc, test_acc



if __name__ == "__main__":
    main()
