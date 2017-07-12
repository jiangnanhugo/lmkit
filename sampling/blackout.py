import numpy as np
import theano
import theano.tensor as T



class Blackout(object):
    # noise contrastive estimation version output probability
    def __init__(self,n_input,n_output,x,y,y_mask,y_neg,q_w=None,k=10):
        self.q_w=q_w
        self.k=k

        self.x = x.reshape([-1, x.shape[-1]])
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
        self.build()


    def build(self):
        # blackout version output probability
        # correct word probability (1,1)
        c_o_t = T.exp(V[y_t].dot(s_t) + c[y_t])

        # negative word probability (k,1)
        n_o_t = T.exp(V[neg_y_t].dot(s_t) + c[neg_y_t])

        # sample set probability
        t_o = (q_w[y_t] * c_o_t) + T.sum(q_w[neg_y_t] * n_o_t)

        # positive probability
        c_o_p = q_w[y_t] * c_o_t / t_o

        # negative probability (k,1)
        n_o_p = q_w[neg_y_t] * n_o_t / t_o

        # cost for each y in blackout
        J_dis = -(T.log(c_o_p) + T.sum(T.log(T.ones_like(n_o_p) - n_o_p)))


        # cost for each y in nce
        self.activation = -T.mean(T.log(c_o_p) + T.sum(T.log(n_o_p)))
        att = T.nnet.softmax(T.dot(self.x, self.W) + self.b)
        self.predict = T.argmax(att, axis=-1)

