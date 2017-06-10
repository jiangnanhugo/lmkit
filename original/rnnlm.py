import theano
if theano.config.device=='cpu':
    from theano.tensor.shared_randomstreams import RandomStreams
else:
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from softmax import softmax
from layers.gru import GRU
from layers.lstm import LSTM
from layers.updates import *

class RNNLM(object): 
    def __init__(self, n_input, n_hidden, n_output, cell='gru', optimizer='sgd', p=0.5): 
        self.x = T.imatrix('batched_sequence_x')  # n_batch, maxlen 
        self.x_mask = T.matrix('x_mask') 
        self.y = T.imatrix('batched_sequence_y') 
        self.y_mask = T.matrix('y_mask') 
        self.n_input = n_input 
        self.n_hidden = n_hidden 
        self.n_output = n_output 

        init_Embd = np.asarray(np.random.uniform(low=-np.sqrt(1. / n_output), high=np.sqrt(1. / n_output), size=(n_output, n_input)), dtype=theano.config.floatX) 

        self.E = theano.shared(value=init_Embd, name='word_embedding',borrow=True) 
        self.cell = cell 
        self.optimizer = optimizer 
        self.p = p

        
        self.is_train = T.iscalar('is_train')

        self.rng = RandomStreams(1234)
        self.build()

    def build(self):
        print 'building rnn cell...'
        if self.cell == 'gru':
            hidden_layer = GRU(self.rng,
                               self.n_input, self.n_hidden, 
                               self.x, self.E, self.x_mask,
                               self.is_train, self.p)
        else:
            hidden_layer = LSTM(self.rng,
                                self.n_input, self.n_hidden, 
                                self.x, self.E, self.x_mask,
                                self.is_train, self.p)
        print 'building softmax output layer...'
        output_layer = softmax(self.n_hidden, self.n_output, hidden_layer.activation)
        cost= self.categorical_crossentropy(output_layer.activation, self.y) # nll
        self.params = [self.E, ]
        self.params += hidden_layer.params
        self.params += output_layer.params


        lr = T.scalar("lr")
        gparams = [T.clip(T.grad(cost, p), -10, 10) for p in self.params]

        if self.optimizer =='sgd':
            updates = sgd(self.params, gparams, lr)
        elif self.optimizer == 'adam':
            updates=adam(self.params,gparams,lr)
        elif self.optimizer =='rmsprop':
            updates=rmsprop(params=self.params,grads=gparams,learning_rate=lr)

        self.train = theano.function(inputs=[self.x, self.x_mask, self.y, self.y_mask, lr],
                                     outputs=cost,
                                     updates=updates,
                                     givens={self.is_train: np.cast['int32'](1)})
        '''
        self.predict=theano.function(inputs=[self.x,self.x_mask],
                                     outputs=output_layer.predict,
                                     givens={self.is_train:np.cast['int32'](1)})
        '''
        self.test = theano.function(inputs=[self.x, self.x_mask,self.y,self.y_mask],
                                       outputs=cost, #nll
                                       givens={self.is_train: np.cast['int32'](0)})

    def categorical_crossentropy(self, y_pred, y_true):
        y_true = y_true.flatten()
        mask=self.y_mask.flatten()
        nll = T.nnet.categorical_crossentropy(y_pred, y_true)
        batch_nll=T.sum(nll*mask)
        return batch_nll/ T.sum(mask)# ,batch_nll
