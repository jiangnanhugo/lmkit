from lmkit.layers.gru import GRU
from lmkit.layers.FastGRU import FastGRU
from lmkit.layers.FastLSTM import FastLSTM
from lmkit.layers.gru import GRU
from lmkit.layers.lstm import LSTM
from lmkit.updates import *
from c_softmax import C_softmax
import cPickle as pickle

if theano.config.device == 'cpu':
    from theano.tensor.shared_randomstreams import RandomStreams
else:
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


class RNNLM(object):
    def __init__(self, n_input, n_hidden, n_output, rnn_cell='gru', optimizer='sgd', p=0.5, n_class=10, node_maxlen=10,node_mask_path=None):
        self.x = T.imatrix('batched_sequence_x')  # [n_batch, maxlen]: int32 matrix
        self.x_mask = T.fmatrix('x_mask')

        self.y_node = T.imatrix('batched_node_y')  # [2, maxlen*batch_size], 2 stands for (class_id,word_id)

        self.y_mask = T.fvector('y_mask')

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_class = n_class
        self.n_output = n_output
        init_Embd = np.asarray(np.random.uniform(low=-np.sqrt(6. / (n_output + n_input)),
                                                 high=np.sqrt(6. / (n_output + n_input)),
                                                 size=(n_output, n_input)),
                               dtype=theano.config.floatX)
        self.E = theano.shared(value=init_Embd, name='word_embedding', borrow=True)

        self.rnn_cell = rnn_cell
        self.optimizer = optimizer
        self.p = p
        self.is_train = T.iscalar('is_train')

        self.rng = RandomStreams(1234)

        self.y_node_mask = pickle.load(open(node_mask_path)) # [2, maxlen*batch_size] 2 stands for (start_index, end_index)
        self.node_maxlen=node_maxlen
        self.build()

    def build(self):
        print 'building rnn cell...'
        hidden_layer=None
        if self.rnn_cell == 'gru':
            hidden_layer = GRU(self.rng,
                               self.n_input, self.n_hidden,
                               self.x, self.E, self.x_mask,
                               self.is_train, self.p)
        elif self.rnn_cell == 'fastgru':
            hidden_layer = FastGRU(self.rng,
                                   self.n_input, self.n_hidden,
                                   self.x, self.E, self.x_mask,
                                   self.is_train, self.p)
        elif self.rnn_cell == 'lstm':
            hidden_layer = LSTM(self.rng,
                                self.n_input, self.n_hidden,
                                self.x, self.E, self.x_mask,
                                self.is_train, self.p)
        elif self.rnn_cell == 'fastlstm':
            hidden_layer = FastLSTM(self.rng,
                                    self.n_input, self.n_hidden,
                                    self.x, self.E, self.x_mask,
                                    self.is_train, self.p)
        print 'building softmax output layer...'
        softmax_shape = (self.n_hidden, self.n_class, self.n_output)
        output_layer = C_softmax(softmax_shape,
                                 hidden_layer.activation,
                                 self.y_node, self.y_node_mask,self.node_maxlen)
        cost = self.categorical_crossentropy(output_layer.activation)

        self.params = [self.E, ]
        self.params += hidden_layer.params
        self.params += output_layer.params

        lr = T.scalar("lr")
        gparams = [T.clip(T.grad(cost, p), -10, 10) for p in self.params]
        updates=None
        if self.optimizer =='sgd':
            updates = sgd(self.params, gparams, lr)
        elif self.optimizer == 'adam':
            updates=adam(self.params,gparams,lr)
        elif self.optimizer =='rmsprop':
            updates=rmsprop(params=self.params,grads=gparams,learning_rate=lr)

        self.train = theano.function(inputs=[self.x, self.x_mask, self.y_node, self.y_mask, lr],
                                     outputs=[cost,output_layer.node],
                                     updates=updates,
                                     givens={self.is_train: np.cast['int32'](1)})
        '''
        self.predict = theano.function(inputs=[self.x, self.x_mask],
                                       outputs=output_layer.predicted,
                                       givens={self.is_train: np.cast['int32'](0)})
        self.test = theano.function(inputs=[self.x, self.x_mask, self.y_node, self.y_node_mask, self.y_mask],
                                    outputs=cost,
                                    givens={self.is_train: np.cast['int32'](0)})
        '''

    def categorical_crossentropy(self, y_pred):
        return -T.sum(y_pred * self.y_mask.flatten()) / T.sum(self.y_mask)