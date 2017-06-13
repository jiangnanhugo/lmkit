import numpy as np
import theano
import theano.tensor as T


class FastLSTM(object):
    """
    LSTM with faster implementatin.
    """

    def __init__(self,rng,n_input,n_hidden,
                 x,E,mask,is_train=1,p=0.5):
        self.rng=rng

        self.n_input=n_input
        self.n_hidden=n_hidden

        self.x=x
        self.E=E
        self.mask=mask
        self.is_train=is_train
        self.p=p
        self.f=T.nnet.sigmoid

        init_W=np.asarray(np.random.uniform(low=-np.sqrt(1./n_input),
                                             high=np.sqrt(1./n_input),
                                             size=(n_input,n_hidden*4)),
                           dtype=theano.config.floatX)
        init_U = np.asarray(np.random.uniform(low=-np.sqrt(1. / n_input),
                                              high=np.sqrt(1. / n_input),
                                              size=(n_hidden, n_hidden * 4)),
                            dtype=theano.config.floatX)

        init_b=np.zeros((n_hidden*4),dtype=theano.config.floatX)

        self.W=theano.shared(value=init_W,name="W")
        self.U = theano.shared(value=init_U, name="U")
        self.b = theano.shared(value=init_b, name="b")

        self.params=[self.W,self.U,self.b]

        self.build()



    def build(self):
        def split(x,n,dim):
            return x[:,n*dim:(n+1)*dim]

        def __recurrence(x_t,m,h_tm1,c_tm1):
            p=x_t+T.dot(h_tm1,self.U)
            # Input Gate
            i_t = self.f(split(p,0,self.n_hidden))
            # Forget Gate
            f_t = self.f(split(p,1,self.n_hidden))
            # Output Gate
            o_t = self.f(split(p,2,self.n_hidden))
            # Cell update
            c_tilde_t=T.tanh(split(p,3,self.n_hidden))
            c_t=f_t * c_tm1 + i_t * c_tilde_t
            # Hidden State
            h_t = o_t * T.tanh(c_t)

            c_t=c_t * m[:,None]
            h_t=h_t * m[:,None]

            return [h_t,c_t]

        pre=T.dot(self.E[self.x,:],self.W)+self.b

        [h,c],_=theano.scan(fn=__recurrence,
                            sequences=[pre,self.mask],
                            outputs_info=[dict(initial=T.zeros((self.x.shape[-1],self.n_hidden))),
                                          dict(initial=T.zeros((self.x.shape[-1],self.n_hidden)))])
        if self.p>0:
            drop_mask=self.rng.binomial(n=1,p=1-self.p,size=h.shape,dtype=theano.config.floatX)
            self.activation=T.switch(self.is_train,h*drop_mask,h*(1-self.p))
        else:
            self.activation=T.switch(self.is_train,h,h)





