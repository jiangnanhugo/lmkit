import theano
if theano.config.device=='cpu':
    from theano.tensor.shared_randomstreams import RandomStreams
elif theano.config.device=='gpu':
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


from lmkit.layers.gru import GRU
from lmkit.layers.lstm import LSTM


from lmkit.layers.level_softmax import level_softmax
from lmkit.updates import *

class RNNLM(object):
    def __init__(self,n_input,n_hidden,n_output,cell='gru',optimizer='sgd',p=0.5):
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
        if self.cell=='gru':
            hidden_layer=GRU(self.rng,
                             self.n_input,self.n_hidden,self.n_batch,
                             self.x,self.E,self.x_mask,
                             self.is_train,self.p)
        else:
            hidden_layer=LSTM(self.rng,
                              self.n_input,self.n_hidden,self.n_batch,
                              self.x,self.E,self.x_mask,
                              self.is_train,self.p)
        print 'building softmax output layer...'
        output_layer=level_softmax(self.n_hidden,self.n_output,hidden_layer.activation,self.y)
        cost = self.categorical_crossentropy(output_layer.activation)

        self.params=[self.E,]
        self.params+=hidden_layer.params
        self.params+=output_layer.params


        lr=T.scalar("lr")
        gparams=[T.clip(T.grad(cost,p),-10,10) for p in self.params]
        updates=sgd(self.params,gparams,lr)

        self.train=theano.function(inputs=[self.x,self.x_mask,self.y,self.y_mask,self.n_batch,lr],
                                   outputs=cost,
                                   updates=updates,
                                   givens={self.is_train:np.cast['int32'](1)})

        self.predict=theano.function(inputs=[self.x,self.x_mask,self.n_batch],
                                     outputs=output_layer.predicted,
                                     givens={self.is_train:np.cast['int32'](0)})
        self.test = theano.function(inputs=[self.x, self.x_mask, self.y, self.y_mask, self.n_batch],
                                    outputs=cost,
                                    givens={self.is_train: np.cast['int32'](0)})


    def categorical_crossentropy(self,y_pred):
        return -T.sum(T.log(y_pred)*self.y_mask.flatten())/T.sum(self.y_mask)

    def categorical_crossentropy2(self, y_pred, y_true=None):
        nll = T.nnet.categorical_crossentropy(y_pred, y_true.flatten())
        return T.sum(nll * self.y_mask.flatten()) / T.sum(self.y_mask)
    
