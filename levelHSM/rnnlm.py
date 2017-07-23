import theano
if theano.config.device=='cpu':
    from theano.tensor.shared_randomstreams import RandomStreams
else:
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


from lmkit.layers.gru import GRU
from lmkit.layers.FastGRU import FastGRU
from lmkit.layers.lstm import LSTM
from lmkit.layers.FastLSTM import FastLSTM
from lmkit.layers.rnnblock import RnnBlock


from level_softmax import level_softmax
from lmkit.updates import *

class RNNLM(object):
    def __init__(self,n_input,n_hidden,n_output,cell='gru',optimizer='sgd',p=0.5,bptt=-1,level1=-1):
        self.x=T.imatrix('batched_sequence_x')  # n_batch, maxlen
        self.x_mask=T.fmatrix('x_mask')
        self.y=T.imatrix('batched_sequence_y')
        self.y_mask=T.fmatrix('y_mask')
        
        self.n_input=n_input
        self.n_hidden=n_hidden
        self.n_output=n_output
        init_Embd=np.asarray(np.random.uniform(low=-np.sqrt(6./(n_output+n_input)),
                                               high=np.sqrt(6./(n_output+n_input)),
                                               size=(n_output,n_input)),
                           dtype=theano.config.floatX)
        self.E=theano.shared(value=init_Embd,name='word_embedding',borrow=True)

        self.bptt = bptt
        self.level1=level1

        self.cell=cell
        self.optimizer=optimizer
        self.p=p
        self.is_train=T.iscalar('is_train')
        self.n_batch=T.iscalar('n_batch')

        self.epsilon=1.0e-15
        self.rng=RandomStreams(1234)
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
                                   self.is_train, self.p, self.bptt)
        elif self.cell == 'lstm':
            hidden_layer = LSTM(self.rng,
                                self.n_input, self.n_hidden,
                                self.x, self.E, self.x_mask,
                                self.is_train, self.p, self.bptt)
        elif self.cell == 'fastlstm':
            hidden_layer = FastLSTM(self.rng,
                                    self.n_input, self.n_hidden,
                                    self.x, self.E, self.x_mask,
                                    self.is_train, self.p, self.bptt)
        elif self.cell.startswith('rnnblock'):
            mode = self.cell.split('.')[-1]
            hidden_layer = RnnBlock(self.rng,
                                    self.n_hidden, self.x, self.E, self.x_mask, self.is_train, self.p, mode=mode)
        print 'building softmax output layer...'
        if self.level1>1:
            level1 = self.level1
        else:
            level1= int(np.ceil(np.sqrt(self.n_output)))
        output_layer=level_softmax(self.n_hidden,level1,self.n_output,hidden_layer.activation,self.y)
        batch_nll,cost = self.categorical_crossentropy(output_layer.activation)

        self.params=[self.E,]
        self.params+=hidden_layer.params
        self.params+=output_layer.params


        lr=T.scalar("lr")
        gparams=[T.clip(T.grad(cost,p),-10,10) for p in self.params]
        updates=sgd(self.params,gparams,lr)

        self.train=theano.function(inputs=[self.x,self.x_mask,self.y,self.y_mask,lr],
                                   outputs=cost,
                                   updates=updates,
                                   givens={self.is_train:np.cast['int32'](1)})

        self.predict=theano.function(inputs=[self.x,self.x_mask],
                                     outputs=output_layer.predicted,
                                     givens={self.is_train:np.cast['int32'](0)})
        self.test = theano.function(inputs=[self.x, self.x_mask, self.y, self.y_mask],
                                    outputs=[batch_nll,output_layer.prediction],
                                    givens={self.is_train: np.cast['int32'](0)})


    def categorical_crossentropy(self,y_pred):
        batch_nll=-T.sum(T.log(y_pred) * self.y_mask.flatten())
        return batch_nll, batch_nll/T.sum(self.y_mask)

    def categorical_crossentropy2(self, y_pred):
        nll = T.nnet.categorical_crossentropy(y_pred, self.y.flatten())
        return T.sum(nll * self.y_mask.flatten()) / T.sum(self.y_mask)
    
