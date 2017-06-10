import numpy as np
import theano
import theano.tensor as T

class FastGRU(object):

    # https://github.com/nyu-dl/dl4mt-tutorial/tree/master/session3
    def __init__(self,rng,n_input,n_hidden,x,E,mask,is_train=1,p=0.5):
        self.rng=rng

        self.n_input=n_input
        self.n_hidden=n_hidden
        self.f=T.nnet.sigmoid

        self.x=x
        self.E=E
        self.mask=mask
        self.is_train=is_train
        self.p=p

        init_W=np.asarray(np.random.uniform(low=-np.sqrt(1./n_input),
                                            high=np.sqrt(1./n_input),
                                            size=(n_input,n_hidden * 4)),
                          dtype=theano.config.floatX)
        init_U=np.asarray(np.random.uniform(low=-np.sqrt(1./n_input),
                                            high=np.sqrt(1./n_input),
                                            size=(n_input,n_hidden*4)),
                          dtype=theano.config.floatX)
        init_b=np.zeros((n_hidden * 4,),dtype=theano.config.floatX)

        self.W=theano.shared(value=init_W,name='W')
        self.U=theano.shared(value=init_U,name='U')
        self.b=theano.shared(value=init_b,name='b')
        self.params=[self.U,self.U,self.b]
        self.build()

    def build(self):
