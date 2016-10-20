#!/usr/bin/env python2.7
import numpy
import random
import copy
class Param(object):
  def __init__(self, name, min_val, max_val, init_val=None, distrib = 'uniform',scale='log',logbase = numpy.e, interval = None):
    self.name = name
    self.init_val = init_val
    self.min_val = min_val
    self.max_val = max_val
    self.scale = scale # log base is 10
    self.logbase = logbase
    self.param_type = 'continuous'
    self.distrib = distrib
    self.interval = interval
  def __repr__(self):
    return "%s(%f,%f,%s)" % ( self.name,self.min_val,self.max_val,self.scale)

  def get_param_range(self, num_vals,stochastic=False):
    if stochastic:
      if self.distrib == 'normal':
        # bad design but here min_val is mean and max_val is sigma
        val = numpy.random.normal(self.min_val,self.max_val,num_vals)
      else:
        val = numpy.random.rand(num_vals)*(self.max_val - self.min_val) + self.min_val
      if self.scale == "log":
        val = numpy.array([self.logbase ** v for v in val])
    else:
      if self.scale == "log":
        val = numpy.logspace(self.min_val, self.max_val, num_vals,base=self.logbase)
      else:
        val =  numpy.linspace(self.min_val, self.max_val, num_vals)
    if self.interval:
      return (numpy.floor(val / self.interval) * self.interval).astype(int)
    return val
  def get_transformed_param(self, x):
    if self.distrib == 'normal':
      print 'not implemented'
      return None
    else:
      val=x
      if self.scale == "log":
        val = self.logbase**x
      if self.interval:
        val=(numpy.floor(val / self.interval) * self.interval).astype(int)
    return val
  def get_min(self):
    return self.min_val
  def get_max(self):
    return self.max_val
  def get_type(self):
    if self.interval:
      return 'integer'
    return 'continuous'

class IntParam(Param):
  def __init__(self, name, min_val, max_val, init_val=None):
    super(IntParam,self).__init__(name, min_val, max_val, init_val=init_val)
    self.param_type = "integer"

  def get_param_range(self, num_vals,stochastic=False):
    #If num_vals greater than range of integer param then constrain to the range and if stochastic param results in
    #duplicates, only keep unique entrys
    if stochastic:
      return numpy.unique(int( numpy.random.rand(num_vals)*(1 + self.max_val - self.min_val) + self.min_val ))
    return range(self.min_val, self.max_val+1, max(1, (self.max_val-self.min_val)/num_vals))

class CategoricalParam(object):
  def __init__(self, name, val_list, default):
    self.name = name
    self.val_list = val_list
    self.default = default
    self.init_val = default
    self.num_vals = len(self.val_list)
    self.param_type = 'categorical'

  def get_param_range(self, num_vals,stochastic=False):
    if stochastic:
      return [self.val_list[i] for i in numpy.unique(numpy.random.randint(len(self.val_list),size=num_vals))]
    if num_vals >= self.num_vals:
      return self.val_list
    else:
      # return random subset, but include default value
      tmp = list(self.val_list)
      tmp.remove(self.default)
      random.shuffle(tmp)
      return [self.default] + tmp[0:num_vals-1]

class ConditionalParam(object):
  def __init__(self,cond_param,cond_val,param):
    self.name = param.name
    self.cond_param = cond_param
    self.cond_val = cond_val
    self.param = param
    self.param_type = 'conditional'
  def check_condition(self, hps):
    if self.cond_param not in hps:
      return None
    if hps[self.cond_param] == self.cond_val:
      return self.param
    return None


def zoom_space(params,center,pct=0.40):
    new_params=copy.deepcopy(params)
    for p in params.keys():
        range=params[p].max_val-params[p].min_val
        best_val=center[p]
        if params[p].scale=='log':
          best_val=numpy.log(best_val)
        new_min=max(params[p].min_val,best_val-pct/2*range)
        new_max=new_min+(pct*range)
        new_params[p].min_val=new_min
        new_params[p].max_val=new_max

    return new_params

