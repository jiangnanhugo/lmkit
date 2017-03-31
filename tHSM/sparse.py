import theano
import theano.tensor as T
import theano.sparse as S

import scipy.sparse
import numpy
import time
import sys
import random
import numpy.matlib

# ------------------------------------------------------------
# create a sparse matrix
# ------------------------------------------------------------
def sparsify(a, num_in, sb, sc):
    assert type(num_in) == int
    A = a * sb
    from numpy.random import rand, permutation
    for i in range(A.shape[1]):
        perm = permutation(A.shape[0])
        SMALL = perm[num_in:]
        A[SMALL, i] *= sc / sb
    a[:] = A

# ------------------------------------------------------------
def get_function(size = (1000, 1000), times = 100, dtype = theano.config.floatX):

    # PARAMETERS
    learning_rate = 0.001
    num_in = 15
    scale = 1. / 15 ** 0.5
    scale_small = 0

    # ------------------------------------------------------------
    # model parameters
    # ------------------------------------------------------------
    # INPUT/TARGET
    x = T.matrix()
    y = T.ivector()

    # INPUT WEIGHTS
    W_in_dense = numpy.asarray(numpy.matlib.randn(size)).astype(dtype)
    sparsify(W_in_dense, num_in, scale, scale_small)
    W_in_sparse = scipy.sparse.csr_matrix(W_in_dense)

    # INPUT BIAS
    b_in = numpy.zeros(size[0], dtype = dtype)
    b_in_shared = theano.shared(b_in)

    # HIDDEN WEIGHTS
    W_h_dense = numpy.asarray(numpy.matlib.randn(size)).astype(dtype)
    sparsify(W_h_dense, num_in, scale, scale_small)
    W_h_sparse = scipy.sparse.csr_matrix(W_h_dense)

    # HIDDEN BIAS
    b_h = numpy.zeros(size[0], dtype = dtype)
    b_h_shared = theano.shared(b_h)

    # OUTPUT WEIGHTS
    # we don't care about an output layer

    # ------------------------------------------------------------
    # function definitions (std. logistic reg.)
    # ------------------------------------------------------------
    func = {}
    for type_ in ['sparse', 'dense']:

        if(type_ == 'sparse'):
            W_in_shared = theano.shared(W_in_sparse)
            W_h_shared = theano.shared(W_h_sparse)
        else:
            W_in_shared = theano.shared(W_in_dense)
            W_h_shared = theano.shared(W_h_dense)

        params = []
        params.extend([W_in_shared, b_in_shared])
        params.extend([W_h_shared, b_h_shared])

        y_out = T.nnet.sigmoid(theano.dot(x, W_in_shared) + b_in_shared)
        p_y_given_x = T.nnet.softmax(theano.dot(y_out, W_h_shared) + b_h_shared)
        cost = -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y])

        gparams = []
        for param in params:
            gparam = T.grad(cost, param)
            gparams.append(gparam)

        updates = {}
        for param, gparam in zip(params, gparams):
            updates[param] = param - learning_rate * gparam

        grad_func = theano.function(inputs = [x, y], outputs = cost, updates = updates)
        out_func = theano.function(inputs = [x, y], outputs = p_y_given_x, on_unused_input = 'ignore')

        func[type_] = {}
        func[type_]['output'] = out_func
        func[type_]['gradient'] = grad_func

    return func

# ---------------------------------------------------------------------------------
# TEST
# ---------------------------------------------------------------------------------
times = 100
size = (1000, 1000)
dtype = theano.config.floatX
input = numpy.asarray([numpy.matlib.randn(size).astype(dtype) for _ in xrange(times)]).astype(dtype)
target = numpy.asarray([random.randint(0, size[0] - 1) for r in xrange(times * size[0])]).astype('int32')
target = target.reshape(times, size[0])

# fetch functions
func = get_function()

# run the functions
for theano_function in ['output', 'gradient']:
    for type_ in ['sparse', 'dense']:
        start_time = time.clock()
        [func[type_][theano_function](input[i], target[i]) for i in range(times)]
        print '... %s model %s calculation ran for %.2fm' % (type_, theano_function, (time.clock() - start_time) / 60.)
