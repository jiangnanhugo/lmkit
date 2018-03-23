
import theano
import numpy as np
import theano.tensor as T
from theano.gpuarray import dnn
from theano.gpuarray.type import gpuarray_shared_constructor
from theano.gpuarray.tests.config import  mode_with_gpu,test_ctx_name
from theano.gpuarray.tests.rnn_support import LSTM,Model,WrapperLayer

from nose.plugins.skip import SkipTest
import theano.tests.unittest_tools as utt

def test_dnn_rnn_lstm():
    if not dnn.dnn_available(test_ctx_name):
        raise SkipTest(dnn.dnn_available.msg)
    utt.seed_rng()

    # test params
    input_dim = 32
    hidden_dim = 16
    batch_size = 2
    depth = 3
    timesteps = 5

    # test code
    X = T.tensor3('X')
    Y = T.tensor3('Y')
    h0 = T.tensor3('h0')
    c0 = T.tensor3('c0')

    rnnb = dnn.RNNBlock(theano.config.floatX, hidden_dim, depth, 'lstm')
    psize = rnnb.get_param_size([batch_size, input_dim])
    params_cudnn = gpuarray_shared_constructor(
        np.zeros((psize,), dtype=theano.config.floatX))

    model = Model()
    last_layer = WrapperLayer(X)
    last_dim = input_dim
    for i in range(depth):
        lstm = LSTM(last_dim, hidden_dim, last_layer, s0=h0[i, :, :], c0=c0[i, :, :])
        model.add_layer(lstm)
        last_layer = lstm
        last_dim = hidden_dim
        layer_params = lstm.get_params()
        dnn_params = rnnb.split_params(params_cudnn, i,
                                       [batch_size, input_dim])
        for j, p in enumerate(dnn_params):
            p[:] = layer_params[j].get_value(borrow=True,
                                             return_internal_type=True)

    def funcs(out, params):
        fn = theano.function([X, h0, c0], out, mode=mode_with_gpu)
        cost = T.mean((Y - out)**2)
        grad = T.grad(cost, [X, h0, c0] + params)
        grad_fn = theano.function([X, Y, h0, c0], grad, mode=mode_with_gpu)
        return fn, grad_fn

    ref_fn, ref_grad_fn = funcs(last_layer.output(),
                                model.get_params())
    cudnn_fn, cudnn_grad_fn = funcs(rnnb.apply(params_cudnn, X, h0, c0)[0],
                                    [params_cudnn])

    x_val = np.random.random((timesteps, batch_size, input_dim)).astype(theano.config.floatX)
    y_val = np.random.random((timesteps, batch_size, hidden_dim)).astype(theano.config.floatX)
    h0_val = np.random.random((depth, batch_size, hidden_dim)).astype(theano.config.floatX)
    c0_val = np.random.random((depth, batch_size, hidden_dim)).astype(theano.config.floatX)

    ref_out = ref_fn(x_val, h0_val, c0_val)
    cudnn_out = cudnn_fn(x_val, h0_val, c0_val)

    utt.assert_allclose(ref_out, cudnn_out)

    ref_grads = ref_grad_fn(x_val, y_val, h0_val, c0_val)
    cudnn_grads = cudnn_grad_fn(x_val, y_val, h0_val, c0_val)

    utt.assert_allclose(ref_grads[0], cudnn_grads[0])
    utt.assert_allclose(ref_grads[1], cudnn_grads[1])
    utt.assert_allclose(ref_grads[2], cudnn_grads[2])

    ref_grads_params = ref_grads[3:]
    cudnn_grads_params = gpuarray_shared_constructor(cudnn_grads[3])

    for i in range(depth):
        cudnn_grads_layer = rnnb.split_params(cudnn_grads_params, i,
                                              [batch_size, input_dim])
        ref_grads_layer = ref_grads_params[i * len(cudnn_grads_layer):
                                           (i + 1) * len(cudnn_grads_layer)]
        for j, g in enumerate(cudnn_grads_layer):
            utt.assert_allclose(ref_grads_layer[j], g)


test_dnn_rnn_lstm()