import numpy as np
import theano
import theano.tensor as T



class Blackout(object):
    # noise contrastive estimation version output probability
    def __init__(self,n_input,n_output,x,y,y_mask,y_neg,q_w=None,k=10):
        _prefix='blackout_'
        self.q_w=q_w
        self.k=k

        self.x = x.reshape([-1, x.shape[-1]])
        self.y=y.flatten()
        self.y_mask=y_mask.flatten()
        self.y_neg=y_neg.reshape([-1, y_neg.shape[-1]])

        init_W = np.asarray(np.random.uniform(low=-np.sqrt(1. / n_input),
                                              high=np.sqrt(1. / n_input),
                                              size=(n_output, n_input)), dtype=theano.config.floatX)
        init_b = np.zeros((n_output), dtype=theano.config.floatX)

        self.W = theano.shared(value=init_W, name=_prefix+'output_W', borrow=True)
        self.b = theano.shared(value=init_b, name=_prefix+'output_b', borrow=True)

        self.params = [self.W, self.b]
        self.build()


    def build(self):
        # blackout version output probability
        # correct word probability (1,1)
        c_o_t = T.exp(self.W[self.y]*self.x + self.b[self.y])

        # negative word probability (k,1)
        n_o_t = T.exp(self.W[self.y_neg]*self.x.dimshuffle(0,'x',1) + self.b[self.y_neg])

        # sample set probability
        t_o = (self.q_w[self.y] * c_o_t) + T.sum(self.q_w[self.y_neg] * n_o_t)

        # positive probability
        c_o_p = self.q_w[self.y] * c_o_t / t_o

        # negative probability (k,1)
        n_o_p = self.q_w[self.y_neg] * n_o_t / t_o

        # cost for each y in blackout
        self.activation = -(T.log(c_o_p) + T.sum(T.log(T.ones_like(n_o_p) - n_o_p)))
        att = T.nnet.softmax(T.dot(self.x, self.W) + self.b)
        self.predict = T.argmax(att, axis=-1)

