
import numpy as np
import theano
from theano import tensor
from theano.sandbox.cuda import GpuOp
from theano.sandbox.cuda.type import CudaNdarrayType
from theano.sandbox.cuda.basic_ops import (as_cuda_ndarray_variable,
                                           gpu_contiguous)

from theano.gof import COp, Op


def shared_floatx(value, name=None, borrow=False, dtype=None, **kwargs):
    if dtype is None:
        dtype = theano.config.floatX
    return theano.shared(theano._asarray(value, dtype=dtype),
                         name=name, borrow=borrow, **kwargs)

def shared_floatx_zeros(shape, **kwargs):
    return shared_floatx(np.zeros(shape), **kwargs)

class PyCheckGruOp(Op):
    __props__ = ()

    def make_node(self,
            inp_state,
            inp_update,
            inp_reset,

            state_to_state,
            state_to_update,
            state_to_reset
            ):

        weights = [
            state_to_state,
            state_to_update,
            state_to_reset]

        inputs = [inp_state, inp_update, inp_reset]

        for w in weights:
            assert w.dtype == "float32"
            assert w.ndim == 2

        for i in inputs:
            assert i.dtype == "float32"
            assert i.ndim == 3

        out_type = theano.tensor.matrix()
        return theano.Apply(self, inputs+weights, [out_type])

    def perform(self, node, inp, output):
        inp_state, inp_update, inp_reset,\
                state_to_state, state_to_update, state_to_reset \
                = inp
        z, = output

        state = np.zeros((inp_state.shape[1], inp_state.shape[2]), dtype=theano.config.floatX)
        gate_activation=lambda x: 1.0 / (1.0 + np.exp(-x))
        activation=np.tanh

        for i in range(inp_state.shape[0]):
            inp_reset_slice = inp_reset[i]
            inp_update_slice = inp_update[i]
            inp_state_slice = inp_state[i]

            reset_values = gate_activation(
                    state.dot(state_to_reset) + inp_reset_slice)

            update_values = gate_activation(
                    state.dot(state_to_update) + inp_update_slice)

            next_state_proposed = activation(
                (state * reset_values).dot(state_to_state) + inp_state_slice)


            state = (next_state_proposed * update_values +
                           state * (1 - update_values))

        z[0] = state

class PyCheckFactorGruOp(PyCheckGruOp):
    def perform(self, node, inp, output):
        inp_state, inp_update, inp_reset,\
                state_to_state, state_to_update, state_to_reset \
                = inp
        z, = output

        state = np.zeros((inp_state.shape[1], inp_state.shape[2]), dtype=theano.config.floatX)
        gate_activation=lambda x: 1.0 / (1.0 + np.exp(-x))
        activation=np.tanh

        for i in range(inp_state.shape[0]):
            inp_reset_slice = inp_reset[i]
            inp_update_slice = inp_update[i]
            inp_state_slice = inp_state[i]

            A = state.dot(state_to_update) #GEMM
            A = gate_activation(A + inp_update_slice) #ELEM_inplace + 1

            B = state.dot(state_to_reset) #GEMM_inplace +1
            B = state * gate_activation(B + inp_reset_slice) #ELEM

            B = B.dot(state_to_state) #GEMM

            state = activation(B + inp_state_slice) * A + state * (1 - A)

        z[0] = state

class GruOp(GpuOp, COp):
    __props__ = ()

    def __init__(self):
        COp.__init__(self, ['GruOp.cu'],
                'APPLY_SPECIFIC(gated_unit_main_2)')

    def make_node(self,
            initial_state,
            inp_state,
            inp_update,
            inp_reset,

            state_to_state,
            state_to_update,
            state_to_reset
            ):

        weights = [
            state_to_state,
            state_to_update,
            state_to_reset]

        batch_size = inp_state.shape[1]
        assert initial_state.dtype == "float32"
        assert initial_state.ndim == 1

        initial_state = as_cuda_ndarray_variable(
                tensor.repeat(initial_state[None, :], batch_size, 0))

        for i,w in enumerate(weights):
            weights[i] = as_cuda_ndarray_variable(w)

        inputs = [inp_state, inp_update, inp_reset]
        for i,b in enumerate(inputs):
            inputs[i] = as_cuda_ndarray_variable(b)


        for w in weights:
            assert w.dtype == "float32"
            assert w.ndim == 2

        for i in inputs:
            assert i.dtype == "float32"
            assert i.ndim == 3

        out_type = CudaNdarrayType((False, False))
        return theano.Apply(self, [initial_state] + inputs+weights, [out_type()])

    def c_code_cache_version(self):
        return tuple()

gru = GruOp()
gruP = PyCheckGruOp()
gruFac = PyCheckFactorGruOp()


'''
if __name__ == "__main__":
    theano.config.optimizer='None'
    import numpy as np
    from blocks.initialization import IsotropicGaussian, Constant
    x = tensor.tensor3("inp_variable")
    #x = tensor.tensor3("inp_variable")
    n_hid = 512
    n_in = 512

    np.random.seed(1)
    rng = np.random
    init = IsotropicGaussian(0.02)
    #init = Constant(0.00)

    inp_to_state = shared_floatx_zeros((n_in, n_hid))
    init.initialize(inp_to_state, rng)
    inp_to_update = shared_floatx_zeros((n_in, n_hid))
    init.initialize(inp_to_update, rng)
    inp_to_reset = shared_floatx_zeros((n_in, n_hid))
    init.initialize(inp_to_reset, rng)

    inp_to_state_b = shared_floatx_zeros((n_hid,))
    init.initialize(inp_to_state_b, rng)
    inp_to_update_b = shared_floatx_zeros((n_hid,))
    init.initialize(inp_to_update_b, rng)
    inp_to_reset_b = shared_floatx_zeros((n_hid,))
    init.initialize(inp_to_reset_b, rng)

    state_to_state = shared_floatx_zeros((n_hid, n_hid))
    init.initialize(state_to_state, rng)
    state_to_update = shared_floatx_zeros((n_hid, n_hid))
    init.initialize(state_to_update, rng)
    state_to_reset = shared_floatx_zeros((n_hid, n_hid))
    init.initialize(state_to_reset, rng)

    inp_state = tensor.dot(x, inp_to_state) + inp_to_state_b
    inp_update = tensor.dot(x, inp_to_update) + inp_to_update_b
    inp_reset = tensor.dot(x, inp_to_reset) + inp_to_reset_b

    initial_state = shared_floatx_zeros((n_hid, ))
    zeros_state = np.zeros((n_hid, ), dtype=theano.config.floatX)
    initial_state.set_value(zeros_state)


    brick = GatedRecurrentFast()
    brick._params = [None, None, None, None]
    brick.params[0] = state_to_state
    brick.params[1] = state_to_update
    brick.params[2] = state_to_reset
    brick.params[3] = initial_state

    res_blocks = brick.apply(inp_state, inp_update, inp_reset)[-1]

    res = gru(initial_state, inp_state, inp_update, inp_reset,
            state_to_state, state_to_update, state_to_reset)

    resP = gruP(inp_state, inp_update, inp_reset,
            state_to_state, state_to_update, state_to_reset)

    resF = gruFac(inp_state, inp_update, inp_reset,
            state_to_state, state_to_update, state_to_reset)

    inp = np.ones((1, 3, n_in), dtype=theano.config.floatX)
    #f = theano.function([x], res, mode='DebugMode')
    f = theano.function([x], res, name="CustomCudaOp")#, mode='DebugMode')
    fF = theano.function([x], resF, name="RawPython")#, mode='DebugMode')
    f_blocks = theano.function([x], res_blocks, name="BlocksScan")#, mode='DebugMode')
    #print f.maker.fgraph.toposort()
    #print f(np.ones((6, 3, n_in), dtype=theano.config.floatX))

    #theano.printing.pydotprint(f, outfile='custom.png', var_with_name_simple=True, scan_graphs=True)
    #theano.printing.pydotprint(f_blocks, outfile='blocks.png', var_with_name_simple=True, scan_graphs=True)


    print np.asarray(fF(inp))
    print np.asarray(f(inp))
    print np.asarray(f_blocks(inp))

    def TT(inp):
        print "profile"
        import time
        start = time.time()
        [f(inp) for _ in range(30)]
        print time.time()-start
        #start = time.time()
        #[fF(inp) for _ in range(10)]
        #print time.time()-start
        start = time.time()
        [f_blocks(inp) for _ in range(30)]
        print time.time()-start
        pass

    #inp = np.ones((1, 3, n_in), dtype=theano.config.floatX)
    #TT(inp)
    inp = np.ones((100, 64, n_in), dtype=theano.config.floatX)
    #inp = np.ones((1, 3, n_in), dtype=theano.config.floatX)
    TT(inp)
'''
