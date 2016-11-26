#!/bin/bash

sudo apt-get --yes update
sudo apt-get --yes upgrade
sudo apt-get --yes install make
sudo apt-get --yes install wget
sudo apt-get --yes install git
sudo apt-get --yes install screen
sudo apt-get --yes install python-setuptools
sudo apt-get --yes install libpng-dev libjpeg8-dev libfreetype6-dev pkg-config
sudo apt-get --yes install default-jre
sudo apt-get -y install python-pip
sudo apt-get -y install python-tk
sudo pip install virtualenv
sudo apt-get --yes install libblas-dev liblapack-dev libatlas-base-dev gfortran
sudo apt-get --yes install libeigen3-dev
#Install necessary python packages
virtualenv experiments
source experiments/bin/activate
pip install numpy
pip install scipy
pip install xmltodict
pip install scikit-image
sudo apt-get --yes install build-essential
sudo apt-get --yes install python-dev
pip install scikit-learn==0.16.1

git clone https://github.com/jaberg/skdata.git
cd skdata && python setup.py install
cd $HOME

git clone -b jmlr https://github.com/lishal/HPOlib_fork.git
cd HPOlib_fork && python setup.py install
cd $HOME
mv HPOlib_fork HPOlib

git clone https://github.com/lishal/pylearningcurvepredictor.git
cd pylearningcurvepredictor
python setup.py install
cd $HOME
pip install emcee
pip install george
pip install lmfit
pip install triangle

git clone https://github.com/lishal/hyperband_benchmarks.git
echo "/home/ubuntu/HPOlib" > $HOME/experiments/lib/python2.7/site-packages/hpolib.pth


