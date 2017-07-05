import numpy as np
import theano
import theano.tensor as T



class NCE(object):
    # noise contrastive estimation version output probability
    def __init__(self,n_input,n_output,x,y,y_mask,y_neg,q_w=None,k=10):
        self.q_w=q_w
        self.k=k

        self.x=x
        self.y=y
        self.y_mask=y_mask
        self.y_neg=y_neg

        init_W = np.asarray(np.random.uniform(low=-np.sqrt(1. / n_input),
                                              high=np.sqrt(1. / n_input),
                                              size=(n_input, n_output)), dtype=theano.config.floatX)
        init_b = np.zeros((n_output), dtype=theano.config.floatX)

        self.W = theano.shared(value=init_W, name='output_W', borrow=True)
        self.b = theano.shared(value=init_b, name='output_b', borrow=True)

        self.params = [self.W, self.b]


        activation = T.nnet.softmax(T.dot(self.x, self.W) + self.b)
        self.predict = T.argmax(activation, axis=-1)

        self.build()


    def build(self):
        # correct word probability (1,1)
        c_o_t = T.exp(self.W[self.y].dot(self.x) + self.b[self.y])

        # negative word probability (k,1)
        n_o_t = T.exp(self.W[self.y].dot(self.x) + self.b[self.y_neg])

        # positive probability
        c_o_p = c_o_t / (c_o_t + self.k * self.q_w[self.y])

        # negative probability (k,1)
        n_o_p = self.q_w[self.y_neg] / (n_o_t + self.k * self.q_w[self.y_neg])

        # cost for each y in nce
        self.activation = -(T.log(c_o_p) + T.sum(T.log(n_o_p)))

