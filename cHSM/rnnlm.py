import theano
from lmkit.layers.gru import GRU
from lmkit.layers.FastGRU import FastGRU
from lmkit.layers.lstm import LSTM
from lmkit.layers.FastLSTM import FastLSTM
from lmkit.updates import *
from lmkit.layers.c_softmax import c_softmax
if theano.config.device == 'cpu':
    from theano.tensor.shared_randomstreams import RandomStreams
elif theano.config.device == 'gpu':
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams



class RNNLM(object):
    def __init__(self,n_input,n_hidden,n_output,n_class,cell='gru',optimizer='sgd',p=0.5):
        self.x=T.imatrix('batched_sequence_x')  # [n_batch, maxlen]: int32 matrix
        self.x_mask=T.fmatrix('x_mask')

        self.y_node=T.itensor3('batched_node_y') #  [2, maxlen, batch_size], 2 stands for (class_id,word_id)
        self.y_arange_cache=T.itensor3('batched_choice_y') # [2, maxlen, batch_size] 2 stands for (start_index, end_index)
        self.y_mask=T.fmatrix('y_mask')

        self.n_input=n_input
        self.n_hidden=n_hidden
        self.n_class=n_class
        self.n_output=n_output
        init_Embd=np.asarray(np.random.uniform(low=-np.sqrt(6./(n_output+n_input)),
                                               high=np.sqrt(6./(n_output+n_input)),
                                               size=(n_output,n_input)),
                             dtype=theano.config.floatX)
        self.E=theano.shared(value=init_Embd,name='word_embedding',borrow=True)

        self.cell=cell
        self.optimizer=optimizer
        self.p=p
        self.is_train=T.iscalar('is_train')


        self.rng=RandomStreams(1234)
        self.build()

    def build(self):
        print 'building rnn cell...'
        hidden_layer = None
        if self.cell == 'gru':
            hidden_layer = GRU(self.rng,
                               self.n_input, self.n_hidden,
                               self.x, self.E, self.x_mask,
                               self.is_train, self.p)
        elif self.cell == 'fastgru':
            hidden_layer = FastGRU(self.rng,
                                   self.n_input, self.n_hidden,
                                   self.x, self.E, self.x_mask,
                                   self.is_train, self.p)
        elif self.cell == 'lstm':
            hidden_layer = LSTM(self.rng,
                                self.n_input, self.n_hidden,
                                self.x, self.E, self.x_mask,
                                self.is_train, self.p)
        elif self.cell == 'fastlstm':
            hidden_layer = FastLSTM(self.rng,
                                    self.n_input, self.n_hidden,
                                    self.x, self.E, self.x_mask,
                                    self.is_train, self.p)
        print 'building softmax output layer...'
        softmax_shape=(self.n_hidden,self.n_class,self.n_output)
        output_layer=c_softmax(softmax_shape,
                               hidden_layer.activation,
                               self.y_node,self.y_arange_cache,self.y_mask)
        cost = T.sum(output_layer.activation)

        self.params=[self.E,]
        self.params+=hidden_layer.params
        self.params+=output_layer.params


        lr=T.scalar("lr")
        gparams=[T.clip(T.grad(cost,p),-10,10) for p in self.params]
        updates=sgd(self.params,gparams,lr)

        self.train=theano.function(inputs=[self.x,self.x_mask,self.y_node,self.y_arange_cache,self.y_mask,lr],
                                   outputs=cost,
                                   updates=updates,
                                   givens={self.is_train:np.cast['int32'](1)})

        self.predict=theano.function(inputs=[self.x,self.x_mask],
                                     outputs=output_layer.predicted,
                                     givens={self.is_train:np.cast['int32'](0)})
        self.test = theano.function(inputs=[self.x, self.x_mask, self.y_node,self.y_arange_cache, self.y_mask],
                                    outputs=cost,
                                    givens={self.is_train: np.cast['int32'](0)})

