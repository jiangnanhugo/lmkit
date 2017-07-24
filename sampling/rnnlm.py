import numpy as np
import theano as theano
import theano.tensor as T
from lmkit.layers.softmax import softmax
from lmkit.layers.gru import GRU
from lmkit.layers.FastGRU import FastGRU
from lmkit.layers.lstm import LSTM
from lmkit.layers.FastLSTM import FastLSTM
from lmkit.layers.rnnblock import RnnBlock
from lmkit.updates import *
from nce import NCE
from blackout import Blackout

if theano.config.device=='cpu':
    from theano.tensor.shared_randomstreams import RandomStreams
else:
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
class RNNLM(object):
    def __init__(self,n_input, n_hidden, n_output,
                 cell='gru', optimizer='sgd',p=0.1,
                 q_w=None,k=10,sampling='nce'):
        self.x = T.imatrix('batched_sequence_x')  # n_batch, maxlen
        self.x_mask = T.fmatrix('x_mask')
        self.y = T.imatrix('batched_sequence_y')
        self.y_mask = T.fmatrix('y_mask')
        # negy is the negative sampling for nce shape (len(y),k)
        self.negy = T.itensor3('negy')

        self.n_input=n_input
        self.n_hidden=n_hidden
        self.n_output=n_output

        self.k=k
        self.sampling=sampling
        self.optimizer=optimizer
        self.cell=cell
        self.optimizer=optimizer
        self.p=p
        self.is_train = T.iscalar('is_train')
        self.rng = RandomStreams(1234)


        init_Embd = np.asarray(
            np.random.uniform(low=-np.sqrt(1. / n_output), high=np.sqrt(1. / n_output), size=(n_output, n_input)),
            dtype=theano.config.floatX)

        self.E = theano.shared(value=init_Embd, name='word_embedding', borrow=True)



        self.q_w = theano.shared(value=q_w, name='vocabulary distribution', borrow=True)


        self.build()
    
    def build(self):
        print 'building rnn cell...'
        hidden_layer = None
        if self.cell == 'gru':
            hidden_layer = GRU(self.rng,
                               self.n_input, self.n_hidden,
                               self.x, self.E, self.x_mask,
                               self.is_train, self.p, self.bptt)
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
        elif self.cell == 'rnnblock':
            hidden_layer = RnnBlock(self.rng,
                                    self.n_hidden, self.x, self.E, self.x_mask, self.is_train, self.p)

        print 'building softmax output layer...'
        output_layer=None
        if self.sampling=='nce':
            output_layer = NCE(self.n_hidden, self.n_output,
                           hidden_layer.activation,self.y,self.y_mask,self.negy,
                           q_w=self.q_w,k=self.k)
        elif self.sampling=='blackout':
            output_layer = Blackout(self.n_hidden, self.n_output,
                               hidden_layer.activation, self.y, self.y_mask, self.negy,
                               q_w=self.q_w, k=self.k)


        cost = output_layer.activation
        predicted=output_layer.predict
        nll=self.categorical_crossentropy(output_layer.probability,self.y)


        self.params = [self.E, ]
        self.params += hidden_layer.params
        self.params += output_layer.params

        lr = T.scalar("lr")
        gparams = [T.clip(T.grad(cost, p), -10, 10) for p in self.params]
        updates = None
        if self.optimizer == 'sgd':
            updates = sgd(self.params, gparams, lr)
        elif self.optimizer == 'adam':
            updates = adam(self.params, gparams, lr)
        elif self.optimizer == 'rmsprop':
            updates = rmsprop(params=self.params, grads=gparams, learning_rate=lr)

        
        lr=T.scalar("lr")
        gparams=[T.clip(T.grad(cost,p),-10,10) for p in self.params]

        if self.optimizer =='sgd':
            updates = sgd(self.params, gparams, lr)
        elif self.optimizer == 'adam':
            updates=adam(self.params,gparams,lr)
        elif self.optimizer =='rmsprop':
            updates=rmsprop(params=self.params,grads=gparams,learning_rate=lr)


        self.train=theano.function(inputs=[self.x,self.x_mask,self.y,self.negy,self.y_mask,lr],
                                   outputs=cost,
                                   updates=updates,
                                   givens={self.is_train: np.cast['int32'](1)})
        self.predict = theano.function(inputs=[self.x, self.x_mask],
                                    outputs=predicted,
                                    givens={self.is_train: np.cast['int32'](0)})
        self.test=theano.function(inputs=[self.x,self.x_mask,self.y,self.y_mask],
                                  outputs=[nll,predicted],
                                  givens={self.is_train: np.cast['int32'](0)})

    def categorical_crossentropy(self, y_pred, y_true):
        y_true = y_true.flatten()
        mask=self.y_mask.flatten()
        nll = T.nnet.categorical_crossentropy(y_pred, y_true)

        batch_nll=T.sum(nll*mask)
        return batch_nll
        #return batch_nll/ T.sum(mask)# ,batch_nll
