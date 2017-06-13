import theano
import theano.gradient
import theano.printing
import theano.gof
from theano.sandbox.cuda.basic_ops import (as_cuda_ndarray_variable,gpu_contiguous)

from theano.gof.opt import OpSub
from theano.compile import optdb
import os


class LSTMOpGrad(theano.sanbox.cuda.GpuOp):
    def __init__(self):
        pass