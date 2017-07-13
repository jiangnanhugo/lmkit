import numpy as np
import theano
from theano.tensor.nnet.blocksparse import sparse_block_dot
import theano.tensor as T
import logging



class softmax(object):
    def __init__(self,n_input,n_output,x):
        self.n_input=n_input
        self.n_output=n_output

        self.logit_shape=x.shape
        self.x=x.reshape([self.logit_shape[0]*self.logit_shape[1],self.logit_shape[2]])

        init_W=np.asarray(np.random.uniform(low=-np.sqrt(1./n_input),
                                             high=np.sqrt(1./n_input),
                                             size=(n_input,n_output)),dtype=theano.config.floatX)
        init_b=np.zeros((n_output),dtype=theano.config.floatX)

        self.W=theano.shared(value=init_W,name='output_W')
        self.b=theano.shared(value=init_b,name='output_b')

        self.params=[self.W,self.b]

        self.build()

    def build(self):

        self.activation=T.nnet.softmax(T.dot(self.x,self.W)+self.b)
        self.predict=T.argmax(self.activation,axis=-1)

class level_softmax(object):
    def __init__(self,n_input,n_output,x,y):
        level1_size=int(np.ceil(np.sqrt(n_output)))
        level2_size=int(np.ceil(n_output/(level1_size-1)))
        print "level1_size=%d, level2_size=%d"% (level1_size,level2_size)
        assert level1_size*level2_size>=n_output

        self.logitx_shape=x.shape

        x=x.reshape((-1,self.logitx_shape[-1]))
        y=y.reshape((-1,1))

        init_W1=np.asarray(np.random.uniform(low=-np.sqrt(1./n_input),
                                             high=np.sqrt(1./n_input),
                                             size=(n_input,level1_size)),dtype=theano.config.floatX)
        init_b1=np.zeros((level1_size),dtype=theano.config.floatX)

        init_W2=np.asarray(np.random.uniform(low=-np.sqrt(1./n_input),
                                             high=np.sqrt(1./n_input),
                                             size=(level1_size,n_input,level2_size)),dtype=theano.config.floatX)
        init_b2=np.zeros((level1_size,level2_size),dtype=theano.config.floatX)

        self.W1=theano.shared(value=init_W1,name='output_W1',borrow=True)
        self.b1=theano.shared(value=init_b1,name='output_b1',borrow=True)
        self.W2=theano.shared(value=init_W2,name='output_W2',borrow=True)
        self.b2=theano.shared(value=init_b2,name='output_b2',borrow=True)

        self.params=[self.W1,self.b1,self.W2,self.b2]


        self.activation = h_softmax(x,x.shape[0], n_output, level1_size,level2_size,
                                    self.W1, self.b1, self.W2, self.b2,
                                    y)

        self.predicted=h_softmax(x, x.shape[0], n_output, level1_size,level2_size,
                                 self.W1, self.b1, self.W2, self.b2)

        self.prediction=T.argmax(self.predicted,axis=-1)




def h_softmax(x, batch_size, n_outputs, n_classes, n_outputs_per_class,
              W1, b1, W2, b2, target=None):
    """ Two-level hierarchical softmax.

    The architecture is composed of two softmax layers: the first predicts the
    class of the input x while the second predicts the output of the input x in
    the predicted class.
    More explanations can be found in the original paper [1]_.

    If target is specified, it will only compute the outputs of the
    corresponding targets. Otherwise, if target is None, it will compute all
    the outputs.

    The outputs are grouped in the same order as they are initially defined.

    .. versionadded:: 0.7.1

    Parameters
    ----------
    x: tensor of shape (batch_size, number of features)
        the minibatch input of the two-layer hierarchical softmax.
    batch_size: int
        the size of the minibatch input x.
    n_outputs: int
        the number of outputs.
    n_classes: int
        the number of classes of the two-layer hierarchical softmax. It
        corresponds to the number of outputs of the first softmax. See note at
        the end.
    n_outputs_per_class: int
        the number of outputs per class. See note at the end.
    W1: tensor of shape (number of features of the input x, n_classes)
        the weight matrix of the first softmax, which maps the input x to the
        probabilities of the classes.
    b1: tensor of shape (n_classes,)
        the bias vector of the first softmax layer.
    W2: tensor of shape (n_classes, number of features of the input x, n_outputs_per_class)
        the weight matrix of the second softmax, which maps the input x to
        the probabilities of the outputs.
    b2: tensor of shape (n_classes, n_outputs_per_class)
        the bias vector of the second softmax layer.
    target: tensor of shape either (batch_size,) or (batch_size, 1)
        (optional, default None)
        contains the indices of the targets for the minibatch
        input x. For each input, the function computes the output for its
        corresponding target. If target is None, then all the outputs are
        computed for each input.

    Returns
    -------
    output_probs: tensor of shape (batch_size, n_outputs) or (batch_size, 1)
        Output of the two-layer hierarchical softmax for input x. If target is
        not specified (None), then all the outputs are computed and the
        returned tensor has shape (batch_size, n_outputs). Otherwise, when
        target is specified, only the corresponding outputs are computed and
        the returned tensor has thus shape (batch_size, 1).

    Notes
    -----
    The product of n_outputs_per_class and n_classes has to be greater or equal
    to n_outputs. If it is strictly greater, then the irrelevant outputs will
    be ignored.
    n_outputs_per_class and n_classes have to be the same as the corresponding
    dimensions of the tensors of W1, b1, W2 and b2.
    The most computational efficient configuration is when n_outputs_per_class
    and n_classes are equal to the square root of n_outputs.

    References
    ----------
    .. [1] J. Goodman, "Classes for Fast Maximum Entropy Training,"
        ICASSP, 2001, <http://arxiv.org/abs/cs/0108006>`.
    """

    # First softmax that computes the probabilities of belonging to each class
    class_probs = theano.tensor.nnet.softmax(T.dot(x, W1) + b1)

    if target is None:  # Computes the probabilites of all the outputs

        # Second softmax that computes the output probabilities
        activations = T.tensordot(x, W2, (1, 1)) + b2
        output_probs = theano.tensor.nnet.softmax(
            activations.reshape((-1, n_outputs_per_class)))
        output_probs = output_probs.reshape((batch_size, n_classes, -1))
        output_probs = class_probs.dimshuffle(0, 1, 'x') * output_probs
        output_probs = output_probs.reshape((batch_size, -1))
        # output_probs.shape[1] is n_classes * n_outputs_per_class, which might
        # be greater than n_outputs, so we ignore the potential irrelevant
        # outputs with the next line:
        output_probs = output_probs[:, :n_outputs]

    else:  # Computes the probabilities of the outputs specified by the targets

        target = target.flatten()

        # Classes to which belong each target
        target_classes = target // n_outputs_per_class

        # Outputs to which belong each target inside a class
        target_outputs_in_class = target % n_outputs_per_class

        # Second softmax that computes the output probabilities
        activations = sparse_block_dot(
            W2.dimshuffle('x', 0, 1, 2), x.dimshuffle(0, 'x', 1),
            T.zeros((batch_size, 1), dtype='int32'), b2,
            target_classes.dimshuffle(0, 'x'))

        output_probs = theano.tensor.nnet.softmax(activations.dimshuffle(0, 2))
        target_class_probs = class_probs[T.arange(batch_size),
                                         target_classes]
        output_probs = output_probs[T.arange(batch_size),
                                    target_outputs_in_class]
        output_probs = target_class_probs * output_probs

    return output_probs


def test_h_softmax():
    input_size = 4
    batch_size = 2
    h_softmax_level1_size = 101
    h_softmax_level2_size = 100
    output_size = h_softmax_level1_size * h_softmax_level2_size

    #############
    # Initialize shared variables
    #############

    floatX = theano.config.floatX
    shared = theano.shared

    # First level of h_softmax
    W1 = np.asarray(np.random.normal(
        size=(input_size, h_softmax_level1_size)), dtype=floatX)
    W1 = shared(W1)
    b1 = shared(np.asarray(np.zeros((h_softmax_level1_size,)),
                           dtype=floatX))

    # Second level of h_softmax
    W2 = np.asarray(np.random.normal(
        size=(h_softmax_level1_size, input_size, h_softmax_level2_size)),
        dtype=floatX)
    W2 = shared(W2)
    b2 = shared(
        np.asarray(np.zeros((h_softmax_level1_size,
                             h_softmax_level2_size)), dtype=floatX))

    #############
    # Build graph
    #############
    x = T.matrix('x')
    y = T.ivector('y')

    # This only computes the output corresponding to the target
    y_hat_tg = h_softmax(x, batch_size, output_size, h_softmax_level1_size,
                         h_softmax_level2_size, W1, b1, W2, b2, y)

    # This computes all the outputs
    y_hat_all = h_softmax(x, batch_size, output_size, h_softmax_level1_size,
                          h_softmax_level2_size, W1, b1, W2, b2)

    #############
    # Compile functions
    #############
    fun_output_tg = theano.function([x, y], y_hat_tg)
    fun_output = theano.function([x], y_hat_all)

    #############
    # Test
    #############
    x_mat = np.random.normal(size=(batch_size, input_size)).astype(floatX)
    y_mat = np.random.randint(0, output_size, batch_size).astype('int32')
    tg_output = fun_output_tg(x_mat, y_mat)
    print tg_output
    all_outputs = fun_output(x_mat)
    print all_outputs

    assert (tg_output.shape == (batch_size,))
    assert (all_outputs.shape == (batch_size, output_size))

    # Verifies that the outputs computed by fun_output_tg are the same as those
    # computed by fun_output.
    # utt.assert_allclose(all_outputs[np.arange(0, batch_size), y_mat], tg_output)


if __name__=="__main__":
    test_h_softmax()