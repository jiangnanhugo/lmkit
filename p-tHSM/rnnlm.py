from theano.tensor.shared_randomstreams import RandomStreams

from h_softmax import H_Softmax
from gru import GRU
from lstm import LSTM
from updates import *

class RNNLM(object):
    def __init__(self,n_input,n_hidden,n_output,cell='gru',optimizer='sgd',p=0.5,mode='vector'):
        self.x=T.imatrix('batched_sequence_x')  # n_batch, maxlen
        self.x_mask=T.fmatrix('x_mask')
        self.y_node=T.itensor3('batched_node_y')
        self.y_choice=T.ftensor3('batched_choice_y')
        self.y_bit_mask=T.ftensor3('batched_bit_mask_y')
        self.y_mask=T.fmatrix('y_mask')
        
        self.n_input=n_input
        self.n_hidden=n_hidden
        self.n_output=n_output
        init_Embd=np.asarray(np.random.uniform(low=-np.sqrt(1./n_output),
                                               high=np.sqrt(1./n_output),
                                               size=(n_output,n_input)),
                           dtype=theano.config.floatX)
        self.E=theano.shared(value=init_Embd,name='word_embedding')

        self.cell=cell
        self.optimizer=optimizer
        self.p=p
        self.mode=mode
        self.is_train=T.iscalar('is_train')
        self.n_batch=T.iscalar('n_batch')

        self.epsilon=1.0e-15
        self.rng=RandomStreams(1234)
        self.build()

    def build(self):
        print '\t building rnn cell...'
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
        print '\t building softmax output layer...'
        softmax_shape=(self.n_hidden,self.n_output)
        output_layer=H_Softmax(softmax_shape,
                               hidden_layer.activation,
                               self.y_node,self.y_choice,self.y_bit_mask,self.y_mask,mode=self.mode)
        self.params=[self.E,]
        self.params+=hidden_layer.params
        self.params+=output_layer.params

        cost=output_layer.activation
        lr=T.scalar("lr")
        gparams=[T.clip(T.grad(cost,p),-10,10) for p in self.params]
        updates=sgd(self.params,gparams,lr)

        self.train=theano.function(inputs=[self.x,self.x_mask,self.y_node,self.y_choice,self.y_bit_mask,self.y_mask,self.n_batch,lr],
                                   outputs=cost,
                                   updates=updates,
                                   givens={self.is_train:np.cast['int32'](1)})

        self.test=theano.function(inputs=[self.x,self.x_mask,self.y_node,self.y_choice,self.y_bit_mask,self.y_mask,self.n_batch],
                                   outputs=cost,
                                   givens={self.is_train:np.cast['int32'](0)})
        '''
        self.predict=theano.function(inputs=[self.x,self.x_mask,self.n_batch],
                                     outputs=output_layer.prediction,
                                     givens={self.is_train:np.cast['int32'](0)})
        '''
    
