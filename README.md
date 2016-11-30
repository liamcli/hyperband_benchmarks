# hyperband_benchmarks

This repo contains code to reproduce the experiments in <https://arxiv.org/abs/1603.06560>.

## Step 1 Install necessary software and setup the environment: 
Install CUDA 7.5
Install cudnn 4.0.7

Install Caffe using the github repo with commit sha 389db963419ee8a6af7cc9b7fb39ed2363c83ab5.  Remember to set the use cudnn flag to True and make all.  May need to do a separate make for pycaffe.

Run the hyperband_setup.sh script to get install the packages needed to reproduce the experiments.

## Step 2 Download data and create lmdb datasets for CNN experiments:

1. CNN Experiment:  
  - CIFAR-10: Run the following in the caffe root folder  
	./data/cifar10/get_cifar10.sh  
	./examples/cifar10/create_cifar10.sh  
	Use the make_lmdb.py script in the hyperband_nnet/cifar10 directory to partition in training data into a training and validation set.  Make sure to update the paths in the make_lmdb.py script to your local path for the relevant files.
  - MRBI: Download the "Rotated MNIST digits + background images" dataset from   http://www.iro.umontreal.ca/~lisa/twiki/bin/view.cgi/Public/MnistVariations.  
    NOTE: Make sure you don't download the old version of the data.  Then run the mrbi_to_lmdb.py script in the hyperband_nnet/mrbi folder to create the lmdb dataset.  Make sure to update the paths in the mrbi_to_lmdb.py script to your local path for the relevant files.  Generate the image mean file by running: ./build/tools/compute_image_mean -backend=lmdb &lt;input lmdb folder&gt; &lt;output dir&gt;/mean.binaryproto from the caffe directory.
  - SVHN: Download the 3 mat files from http://ufldl.stanford.edu/housenumbers/.  Then run the mat_to_lmdb.py script in the hyperband_nnet/svhn directory to create the lmdb dataset.  Make sure to update the paths in the mat_to_lmdb.py script to your local path for the relevant files.  Generate the image mean file by running: ./build/tools/compute_image_mean -backend=lmdb &lt;input lmdb folder&gt; &lt;output dir&gt;/mean.binaryproto from the caffe directory.  
2. SVM and Random Features Experiment: Download the python version of the CIFAR-10 data from https://www.cs.toronto.edu/~kriz/cifar.html.

## Step 3 Start running experiments!
HPOlib is used to run SMAC, TPE, Spearmint, and random search.
- Go to directory of benchmark and run HPOlib-run with the desired inputs
- CNN Experiments: For runs that do not use early stopping, set number_of_jobs in the config.cfg file for the benchmark to 50.  If early stopping, set number_of_jobs to 150 as a upper bound and stop a trial early if the total budget of 50R has been exhausted.  Change the data_dir in the corresponding wrapper file to match your local file path for the data.
    - HPOlib-run -o <path to optimizer> -s <seed> --EXPERIMENT:device <GPU number> --EXPERIMENT:do_stop <use early_stopping>
- Kernel Lsqr and Random Features Experiments: Change the data_dir in the config.cfg file to match your local directory for the dataset.
    - HPOlib-run -o <path to optimizer> -s <seed>

Hyperband is in a separate repository.
- To run the hyperband experiments, simply run
- python {hyperband_alg.py, hyperband_inf.py} -m &lt;model&gt; -i &lt;input_dir with data&gt; -o &lt;output_dir&gt; -s &lt;seed&gt; -d &lt;gpu device if relevant else enter anything&gt;
- See the main method of either hyperband_alg.py or hyperband_inf.py to see what models are supported.

The following seeds were used for each dataset and searcher:

CNN CIFAR-10:  
- Hyperband finite seeds: 1400, 2400, 3400, 4400, 5400, 6400, 7400, 8400, 9400, 10400  
- Hyperband inf seeds: 1100, 2100, 3100, 4100, 5100, 6100, 7100, 8100, 9100, 10100  
- Hyperband s4 seeds: 1100, 2100, 3100, 4100, 5100, 6100, 7100, 8100, 9100, 10100  
- SMAC seeds: 2000, 3000, 4000, 5000, 6000, 6700, 7700, 8700, 9700, 10700  
- SMAC_early seeds: 1800, 2000, 2800, 3000, 3800, 4000, 4800, 5000, 5800, 6000  
- TPE seeds: 2000, 3000, 4000, 5000, 6000, 6700, 7700, 8700, 9700, 10700  
- Spearmint seeds: 1000, 1500, 2500, 3500, 4500, 6700, 7700, 8700, 9700, 10700  
- Random seeds: 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11700, 12700, 13700, 14700, 15700, 16700, 17700, 18700, 19700, 20700  

CNN MRBI:  
- Hyperband finite seeds: 1800, 2800, 3800, 4800, 5800, 6800, 7800, 8800, 9800, 10800  
- Hyperband s4 seeds: 1200, 2200, 3200, 4200, 5200, 6200, 7200, 8200, 9200, 10200  
- SMAC seeds: 1700, 2700, 3700, 4700, 5700, 6700, 7700, 8700, 9700, 10700  
- SMAC_early seeds: 1000, 2000, 3000, 4000, 5000, 6700, 7700, 8700, 9700, 10700  
- TPE seeds: 1700, 2700, 3700, 4700, 5700, 6700, 7700, 8700, 9700, 10700  
- Spearmint seeds: 1700, 2700, 3700, 4700, 5700, 6700, 7700, 8700, 9700, 10700  
- Random seeds: 1700, 2700, 3700, 4700, 5700, 6700, 7700, 8700, 9700, 10700, 11700, 12700, 13700, 14700, 15700, 16700, 17700, 18700, 19700, 20700  

CNN SVHN:  
- Hyperband finite seeds: 1900, 2900, 3900, 4900, 5900, 6900, 7900, 8900, 9900, 10900  
- Hyperband s4 seeds: 1300, 2300, 3300, 4300, 5300, 6300, 7300, 8300, 9300, 10300  
- SMAC seeds: 1500, 2500, 3500, 4500, 5500, 6500, 7500, 8500, 9500, 10500  
- SMAC_early seeds: 1600, 2600, 3600, 4600, 5600, 6600, 7600, 8600, 9600, 10600  
- TPE seeds: 1500, 2500, 3500, 4500, 5500, 6500, 7500, 8500, 9500, 10500  
- Spearmint seeds: 1500, 2500, 3500, 4500, 5500, 6700, 7700, 8700, 9700, 10700  
- Random seeds: 1500, 2500, 3500, 4500, 5500, 6500, 7500, 8500, 9500, 10500, 11500, 12500, 13500, 14500, 15500, 16700, 17700, 18700, 19700, 20700  

Kernel Lsqr:  
- Hyperband finite seeds: 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000  
- Hyperband s4 seeds: 1100, 2100, 3100, 4100, 5100, 6100, 7100, 8100, 9100, 10100  
- SMAC seeds: 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000  
- TPE seeds: 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000  
- Random seeds: 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000  

Kernel approximation using random features:  
- Hyperband finite seeds: 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000  
- Hyperband s4 seeds: 1100, 2100, 3100, 4100, 5100, 6100, 7100, 8100, 9100, 10100  
- SMAC seeds: 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000  
- TPE seeds: 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000  
- Spearmint seeds: 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000  
- Random seeds: 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000  


NOTE: For CNN experiments, even though seeds are set, there will still be some randomness due to the implementation for convolution in cudnn.



