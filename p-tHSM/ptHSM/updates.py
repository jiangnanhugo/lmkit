import numpy as np
import theano
import theano.tensor as T

def sgd(params,gparams, lr=0.01):
    return [(p,p-lr*gp)for p,gp in zip(params,gparams)]
