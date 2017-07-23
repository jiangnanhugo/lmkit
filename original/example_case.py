from __future__ import print_function

import theano
import numpy
try:
  import scipy
  import scipy.signal
except ImportError:
  scipy = None
import json
import h5py
import sys
from theano import tensor as T
from theano.tensor.nnet import conv
from theano.ifelse import ifelse
try:
  from theano.tensor.signal import pool
except ImportError:  # old Theano or so...
  pool = None
from NetworkBaseLayer import Layer
from ActivationFunctions import strtoact, strtoact_single_joined, elu
import TheanoUtil
from TheanoUtil import class_idx_seq_to_1_of_k
from Log import log
from cuda_implementation.FractionalMaxPoolingOp import fmp
from math import ceil
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from TheanoUtil import print_to_file, DumpOp


class RNNBlockLayer(ForwardLayer):
  recurrent = True
  layer_class = 'rnnblock'

  def __init__(self, num_layers=1, direction=0, **kwargs):
    # this has to be provided in THEANO_FLAGS as e.g. contexts=gpu0->cuda0
    context_name = kwargs.get('device', str(theano.config.device))
    #if context_name == 'cpu':
    #  context_name = 'gpu0'
    kwargs['device'] = context_name
    #kwargs['n_out'] *= 2
    super(RNNBlockLayer, self).__init__(**kwargs)
    self.params = {}
    #self.attrs['n_out'] /= 2
    #self.set_attr('nout', self.attrs['n_out'] / 4)
    from theano.gpuarray import dnn
    from theano.gpuarray.type import gpuarray_shared_constructor
    from theano.tensor.extra_ops import cpu_contiguous
    #from theano.sandbox.cuda.basic_ops import gpu_contiguous

    rnnb = dnn.RNNBlock(
      dtype=theano.config.floatX,
      hidden_size=self.attrs['n_out'],
      num_layers=num_layers,
      rnn_mode='lstm',
      input_mode='linear',
      direction_mode='unidirectional' if direction != 0 else 'bidirectional',
      context_name=context_name if context_name != 'cpu' else 'gpu0'
      )

    buffer_size = 1 # self.attrs['n_out'] * num_layers
    #X = self.get_linear_forward_output()
    #X = T.concatenate([s.output for s in self.sources],axis=2)[::direction or 1]
    X = cpu_contiguous(T.concatenate([s.output for s in self.sources], axis=2)[::direction or 1])
    #X = cpu_contiguous(self.sources[0].output[::direction or 1])
    #X = T.concatenate([X,T.zeros((X.shape[0],batch_size - X.shape[1] + 1,X.shape[2]),X.dtype)],axis=1)[:,:-1]
    n_in = sum([s.attrs['n_out'] for s in self.sources])
    psize = rnnb.get_param_size([buffer_size, n_in])
    l = numpy.sqrt(6.) / numpy.sqrt(4*self.attrs['n_out'])
    pvalue = numpy.asarray(self.rng.uniform(low=-l, high=l, size=(psize,)), dtype=theano.config.floatX)
    if context_name == 'cpu':
      params_cudnn = self.add_param(self.create_bias(psize,name='cudnn_%s' % self.name))
    else:
      params_cudnn = self.add_param(gpuarray_shared_constructor(pvalue, target=context_name,name='cudnn_%s' % self.name))
    c_init = cpu_contiguous(T.alloc(numpy.cast[theano.config.floatX](0), num_layers, X.shape[1], self.attrs['n_out']))
    h_init = cpu_contiguous(T.alloc(numpy.cast[theano.config.floatX](0), num_layers, X.shape[1], self.attrs['n_out']))

    W_out = self.add_param(self.create_random_uniform_weights(self.attrs['n_out'], self.y_in[self.attrs['target']].n_out))
    b_out = self.add_param(self.create_bias(self.y_in[self.attrs['target']].n_out))

    if context_name == 'cpu':
      self.cost_val = T.constant(0)
      self.error_val = T.constant(0)
      self.known_grads = {}
      return

    out = rnnb.apply(params_cudnn, X, h_init, c_init)[0]
    out = out[::-1]
    out = T.dot(out,W_out) + b_out
    self.y_m = out.reshape((out.shape[0] * out.shape[1],out.shape[2]))

    self.i = (self.index.flatten()>0).nonzero()
    self.y_data_flat = self.y_in[self.attrs['target']].flatten()
    nll, _ = T.nnet.crossentropy_softmax_1hot(x=self.y_m[self.i], y_idx=self.y_data_flat[self.i])
    self.cost_val = T.sum(nll)

    #self.cost_val = -T.sum(T.log(out[:,self.y_in[self.attrs['target']].flatten()][(self.index.flatten()>0).nonzero()]))
    self.known_grads = { params_cudnn : T.grad(self.cost_val, params_cudnn) }
    self.output = out
    self.index = self.sources[0].index

    self.error_val = T.sum(T.neq(T.argmax(self.y_m[self.i], axis=-1), self.y_data_flat[self.i]))

  def cost(self):
    return self.cost_val, self.known_grads

  def errors(self):
    return self.error_val