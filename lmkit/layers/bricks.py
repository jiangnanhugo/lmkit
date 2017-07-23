# -*- coding: utf-8 -*-
import ipdb
from blocks.bricks import Activation, Initializable
from blocks.bricks.recurrent import GatedRecurrent
from blocks.initialization import Constant
from blocks.bricks.base import application, Brick, lazy

from theano import tensor
import numpy as np

from blocks.bricks.recurrent import BaseRecurrent, recurrent
from blocks.bricks import Tanh, Sigmoid
from blocks.utils import shared_floatx_nans, shared_floatx_zeros
from blocks.bricks import Initializable, Sigmoid, Tanh
from blocks.bricks.base import Application, application, Brick, lazy
from blocks.initialization import NdarrayInitialization
from blocks.roles import add_role, WEIGHT, INITIAL_STATE
from blocks.utils import (pack, shared_floatx_nans, shared_floatx_zeros,
                          dict_union, dict_subset, is_shared_variable)
import numpy
import functools
import theano

class GatedRecurrent2(BaseRecurrent, Initializable):
    @lazy(allocation=['dim'])
    def __init__(self, dim, activation=None, gate_activation=None,
                 **kwargs):
        super(GatedRecurrent, self).__init__(**kwargs)
        self.dim = dim

        if not activation:
            activation = Tanh()
        if not gate_activation:
            gate_activation = Sigmoid()
        self.activation = activation
        self.gate_activation = gate_activation

        self.children = [activation, gate_activation]

    @property
    def state_to_state(self):
        return self.params[0]

    @property
    def state_to_gates(self):
        return self.params[1]

    def get_dim(self, name):
        if name == 'mask':
            return 0
        if name in ['inputs', 'states']:
            return self.dim
        if name == 'gate_inputs':
            return 2 * self.dim
        return super(GatedRecurrent, self).get_dim(name)

    def _allocate(self):
        self.params.append(shared_floatx_nans((self.dim, self.dim),
                           name='state_to_state'))
        self.params.append(shared_floatx_nans((self.dim, 2 * self.dim),
                           name='state_to_gates'))
        self.params.append(shared_floatx_zeros((self.dim,),
                           name="initial_state"))
        for i in range(2):
            if self.params[i]:
                add_role(self.params[i], WEIGHT)
        add_role(self.params[2], INITIAL_STATE)

    def _initialize(self):
        self.weights_init.initialize(self.state_to_state, self.rng)
        state_to_update = self.weights_init.generate(
            self.rng, (self.dim, self.dim))
        state_to_reset = self.weights_init.generate(
            self.rng, (self.dim, self.dim))
        self.state_to_gates.set_value(
            numpy.hstack([state_to_update, state_to_reset]))


    #@recurrent(sequences=['mask', 'inputs', 'gate_inputs'],
               #states=['states'], outputs=['states'], contexts=[])
    def apply(self, inputs, gate_inputs, mask=None):
        def step(inputs, gate_inputs, states, state_to_gates, state_to_state):
            #import ipdb
            #ipdb.set_trace()
            gate_values = self.gate_activation.apply(
                states.dot(self.state_to_gates) + gate_inputs)
            update_values = gate_values[:, :self.dim]
            reset_values = gate_values[:, self.dim:]
            states_reset = states * reset_values
            next_states = self.activation.apply(
                states_reset.dot(self.state_to_state) + inputs)
            next_states = (next_states * update_values +
                           states * (1 - update_values))
            return next_states

        def step_mask(inputs, gate_inputs, mask_input, states, state_to_gates, state_to_state):
            next_states = step(inputs, gate_inputs, states, state_to_gates, state_to_state)
            if mask_input:
                next_states = (mask_input[:, None] * next_states +
                               (1 - mask_input[:, None]) * states)
            return next_states


        if mask:
            func = step_mask
            sequences = [inputs, gate_inputs, mask]
        else:
            func = step
            sequences = [inputs, gate_inputs]
        #[dict(input=inputs), dict(input=gate_inputs), dict(input=mask)]
        output = tensor.repeat(self.params[2].dimshuffle('x',0), inputs.shape[1], axis=0)
        states_output, _ = theano.scan(fn=func,
                sequences=sequences,
                outputs_info=[output],
                non_sequences=[self.state_to_gates, self.state_to_state],
                strict=True,
                #allow_gc=False)
                )

        return states_output

    @application
    def initial_state(self, state_name, batch_size, *args, **kwargs):
        return tensor.repeat(self.params[2][None, :], batch_size, 0)

class GatedRecurrentFast(BaseRecurrent, Initializable):
    @lazy(allocation=['dim'])
    def __init__(self, dim, activation=None, gate_activation=None,
                 **kwargs):
        super(GatedRecurrentFast, self).__init__(**kwargs)
        self.dim = dim

        if not activation:
            activation = Tanh()
        if not gate_activation:
            gate_activation = Sigmoid()
        self.activation = activation
        self.gate_activation = gate_activation

        self.children = [activation, gate_activation]

    @property
    def state_to_state(self):
        return self.params[0]

    @property
    def state_to_update(self):
        return self.params[1]

    @property
    def state_to_reset(self):
        return self.params[1]

    def get_dim(self, name):
        if name == 'mask':
            return 0
        if name in ['inputs', 'states']:
            return self.dim
        if name in ['update_inputs', 'reset_inputs']:
            return self.dim
        return super(GatedRecurrent, self).get_dim(name)

    def _allocate(self):
        self.params.append(shared_floatx_nans((self.dim, self.dim),
                           name='state_to_state'))
        self.params.append(shared_floatx_nans((self.dim, self.dim),
                           name='state_to_update'))
        self.params.append(shared_floatx_nans((self.dim, self.dim),
                           name='state_to_reset'))

        self.params.append(shared_floatx_zeros((self.dim,),
                           name="initial_state"))
        for i in range(3):
            if self.params[i]:
                add_role(self.params[i], WEIGHT)
        add_role(self.params[3], INITIAL_STATE)

    def _initialize(self):
        self.weights_init.initialize(self.state_to_state, self.rng)
        self.weights_init.initialize(self.state_to_update, self.rng)
        self.weights_init.initialize(self.state_to_reset, self.rng)

    #@recurrent(sequences=['mask', 'inputs', 'gate_inputs'],
               #states=['states'], outputs=['states'], contexts=[])
    def apply(self, inputs, update_inputs, reset_inputs, mask=None):
        def step(inputs, update_inputs, reset_inputs, states, state_to_update, state_to_reset, state_to_state):
            #import ipdb
            #ipdb.set_trace()
            reset_values = self.gate_activation.apply(
                    states.dot(self.state_to_reset) + reset_inputs)

            update_values = self.gate_activation.apply(
                    states.dot(self.state_to_update) + update_inputs)

            next_states_proposed = self.activation.apply(
                (states * reset_values).dot(self.state_to_state) + inputs)

            next_states = (next_states_proposed * update_values +
                           states * (1 - update_values))
            return next_states

        def step_mask(inputs, update_inputs, reset_inputs, mask_input, states, state_to_update, state_to_reset, state_to_state):
            next_states = step(inputs, updatE_inputs, reset_inputs, states, state_to_update, state_to_reset, state_to_state)
            if mask_input:
                next_states = (mask_input[:, None] * next_states +
                               (1 - mask_input[:, None]) * states)
            return next_states


        if mask:
            func = step_mask
            sequences = [inputs, update_inputs, reset_inputs, mask]
        else:
            func = step
            sequences = [inputs, update_inputs, reset_inputs]
        #[dict(input=inputs), dict(input=gate_inputs), dict(input=mask)]
        #output = tensor.repeat(self.params[2].dimshuffle('x',0), inputs.shape[1], axis=0)
        states_output, _ = theano.scan(fn=func,
                sequences=sequences,
                outputs_info=[self.initial_state('initial_state', inputs.shape[1])],
                non_sequences=[self.state_to_reset, self.state_to_update, self.state_to_state],
                strict=True,
                allow_gc=False)

        return states_output

    @application
    def initial_state(self, state_name, batch_size, *args, **kwargs):
        return tensor.repeat(self.params[3][None, :], batch_size, 0)


class WeightedSigmoid(Activation):
    """Weighted sigmoid
    f(x) = 1.0/ (1.0 + exp(-a * x))

    Parameters
    ----------
    a : float

    References
    ---------
    .. [1] Qi Lyu, Jun Zhu
           "Revisit LongShort-Term Memory: An Optimization Perspective"
    """
    def __init__(self, a=1.0, **kwargs):
        self.a = a
        super(WeightedSigmoid, self).__init__(**kwargs)

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return 1.0 / (1.0 + tensor.exp(- self.a * input_))

class WeightedSigmoid(Activation):
    """Weighted sigmoid
    f(x) = 1.0/ (1.0 + exp(-a * x))

    Parameters
    ----------
    a : float

    References
    ---------
    .. [1] Qi Lyu, Jun Zhu
           "Revisit LongShort-Term Memory: An Optimization Perspective"
    """
    def __init__(self, a=1.0, **kwargs):
        self.a = a
        super(WeightedSigmoid, self).__init__(**kwargs)

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return 1.0 / (1.0 + tensor.exp(- self.a * input_))


class GatedRecurrentFull(Initializable):
    """A wrapper around the GatedRecurrent brick that improves usability.
    It contains:
        * A fork to map to initialize the reset and the update units.
        * Better initialization to initialize the different pieces
    While this works, there is probably a better more elegant way to do this.

    Parameters
    ----------
    hidden_dim : int
        dimension of the hidden state
    activation : :class:`.Brick`
    gate_activation: :class:`.Brick`

    state_to_state_init: object
        Weight Initialization
    state_to_reset_init: object
        Weight Initialization
    state_to_update_init: obje64
        Weight Initialization

    input_to_state_transform: :class:`.Brick`
        [CvMG14] uses Linear transform
    input_to_reset_transform: :class:`.Brick`
        [CvMG14] uses Linear transform
    input_to_update_transform: :class:`.Brick`
        [CvMG14] uses Linear transform

    References
    ---------
        self.rnn = GatedRecurrent(
                weights_init=Constant(np.nan),
                dim=self.hidden_dim,
                activation=self.activation,
                gate_activation=self.gate_activation)
    .. [CvMG14] Kyunghyun Cho, Bart van Merriënboer, Çağlar Gülçehre,
        Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua
        Bengio, *Learning Phrase Representations using RNN Encoder-Decoder
        for Statistical Machine Translation*, EMNLP (2014), pp. 1724-1734.

    """
    @lazy(allocation=['hidden_dim', 'state_to_state_init', 'state_to_update_init', 'state_to_reset_init'],
            initialization=['input_to_state_transform', 'input_to_update_transform', 'input_to_reset_transform'])
    def __init__(self, hidden_dim, activation=None, gate_activation=None,
        state_to_state_init=None, state_to_update_init=None, state_to_reset_init=None,
        input_to_state_transform=None, input_to_update_transform=None, input_to_reset_transform=None,
        **kwargs):

        super(GatedRecurrentFull, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim

        self.state_to_state_init = state_to_state_init
        self.state_to_update_init = state_to_update_init
        self.state_to_reset_init = state_to_reset_init

        self.input_to_state_transform = input_to_state_transform
        self.input_to_update_transform = input_to_update_transform
        self.input_to_reset_transform = input_to_reset_transform
        self.input_to_state_transform.name += "_input_to_state_transform"
        self.input_to_update_transform.name += "_input_to_update_transform"
        self.input_to_reset_transform.name += "_input_to_reset_transform"

        self.use_mine = True
        if self.use_mine:
            self.rnn = GatedRecurrentFast(
                    weights_init=Constant(np.nan),
                    dim=self.hidden_dim,
                    activation=activation,
                    gate_activation=gate_activation)
        else:
            self.rnn = GatedRecurrent(
                    weights_init=Constant(np.nan),
                    dim=self.hidden_dim,
                    activation=activation,
                    gate_activation=gate_activation)

        self.children = [self.rnn,
                self.input_to_state_transform, self.input_to_update_transform, self.input_to_reset_transform]
        self.children.extend(self.rnn.children)

    def initialize(self):
        super(GatedRecurrentFull, self).initialize()

        self.input_to_state_transform.initialize()
        self.input_to_update_transform.initialize()
        self.input_to_reset_transform.initialize()

        self.rnn.initialize()

        weight_shape = (self.hidden_dim, self.hidden_dim)
        state_to_state = self.state_to_state_init.generate(rng=self.rng, shape=weight_shape)
        state_to_update= self.state_to_update_init.generate(rng=self.rng, shape=weight_shape)
        state_to_reset = self.state_to_reset_init.generate(rng=self.rng, shape=weight_shape)

        self.rnn.state_to_state.set_value(state_to_state)

        if self.use_mine:
            self.rnn.state_to_update.set_value(state_to_update)
            self.rnn.state_to_reset.set_value(state_to_reset)
        else:
            self.rnn.state_to_gates.set_value(np.hstack((state_to_update, state_to_reset)))

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_, mask=None):
        """

        Parameters
        ----------
        inputs_ : :class:`~tensor.TensorVariable`
            sequence to feed into GRU. Axes are mb, sequence, features

        mask : :class:`~tensor.TensorVariable`
            A 1D binary array with 1 or 0 to represent data given available.

        Returns
        -------
        output: :class:`theano.tensor.TensorVariable`
            sequence to feed out. Axes are batch, sequence, features
        """
        states_from_in = self.input_to_state_transform.apply(input_)
        update_from_in = self.input_to_update_transform.apply(input_)
        reset_from_in = self.input_to_reset_transform.apply(input_)

        gate_inputs = tensor.concatenate([update_from_in, reset_from_in], axis=2)

        if self.use_mine:
            output = self.rnn.apply(inputs=states_from_in, update_inputs=update_from_in, reset_inputs=reset_from_in, mask=mask)
        else:
            output = self.rnn.apply(inputs=states_from_in, gate_inputs=gate_inputs)

        return output


if __name__ == "__main__":
    from blocks.bricks import Linear
    import theano
    floatX = theano.config.floatX
    x = tensor.tensor3('input')
    gru = GatedRecurrentFull(
            hidden_dim=11,
            state_to_state_init=Constant(1.0),
            state_to_reset_init=Constant(1.0),
            state_to_update_init=Constant(1.0),

            input_to_state_transform = Linear(
                input_dim = 19,
                output_dim = 11,
                weights_init=Constant(0.0),
                biases_init=Constant(0.0)
                ),

            input_to_update_transform = Linear(
                input_dim = 19,
                output_dim = 11,
                weights_init=Constant(0.0),
                biases_init=Constant(0.0)
                ),

            input_to_reset_transform = Linear(
                input_dim = 19,
                output_dim = 11,
                weights_init=Constant(0.0),
                biases_init=Constant(0.0)
                ),
            )
    gru.initialize()
    out = gru.apply(x)


    ### Identity testing
    from blocks.initialization import Identity, IsotropicGaussian
    from blocks import bricks
    from blocks.bricks import Sigmoid

    dim = 2
    floatX = theano.config.floatX
    x = tensor.tensor3('input')
    gru = GatedRecurrentFull(
            hidden_dim=dim,
            state_to_state_init=Identity(1.),
            #state_to_reset_init=Identity(1.),
            state_to_reset_init=IsotropicGaussian(0.2),
            state_to_update_init=Identity(1.0),
            activation=bricks.Identity(1.0),
            gate_activation=Sigmoid(),

            input_to_state_transform = Linear(
                input_dim = dim,
                output_dim = dim,
                weights_init=Identity(1.0),
                #weights_init=IsotropicGaussian(0.02),
                biases_init=Constant(0.0)
                ),

            input_to_update_transform = Linear(
                input_dim = dim,
                output_dim = dim,
                #weights_init=Constant(0.0),
                weights_init=IsotropicGaussian(0.02),
                biases_init=Constant(-1)
                ),

            input_to_reset_transform = Linear(
                input_dim = dim,
                output_dim = dim,
                #weights_init=Constant(0.0),
                weights_init=IsotropicGaussian(0.2),
                biases_init=Constant(-100.0)
                ),
            )
    gru.initialize()

    xt = x.dimshuffle(1,0,2)
    #ORDER ON RNN is <sequence, batch, features>
    #xt = x
    out = gru.apply(xt)
    #out = gru.apply(x)
    x_val = np.zeros((1, 6, 2)).astype(dtype=floatX)
    x_val[0][0][0] = 3.0
    #x_val = np.vstack([x_val for _ in range(2)])
    print "Inputting this"
    print x_val
    print "EVALled a thing"
    print xt.eval({x: x_val}).shape, "in shape"
    print out.eval({x: x_val}).shape, "out shape"
    print "Result of rnn"
    print out.eval({x: x_val})
    #print gru.rnn.state_to_state.eval({}), "state to state"

