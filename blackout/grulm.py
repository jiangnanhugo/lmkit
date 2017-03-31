import numpy as np
import theano as theano
import theano.tensor as T
from updates import sgd

class GRULM(object):
    def __init__(self,hidden_dim, word_dim, bptt_truncate=-1):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Initialize the network parameters
        init_E = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.E = theano.shared(value=init_E.astype(theano.config.floatX),name='E')
        init_U = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (6, hidden_dim, hidden_dim))
        self.U = theano.shared(value=init_U.astype(theano.config.floatX),name='U')
        init_W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (6, hidden_dim, hidden_dim))
        self.W = theano.shared(value=init_W.astype(theano.config.floatX),name='W')
        init_V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        self.V = theano.shared(value=init_V.astype(theano.config.floatX),name='V')
        init_b = np.zeros((6, hidden_dim))
        self.b = theano.shared(value=init_b.astype(theano.config.floatX),name='b')
        init_c = np.zeros(word_dim)

        self.c = theano.shared(value=init_c.astype(theano.config.floatX),name='c')

        self.params=[self.E,self.U,self.W,self.V,self.b,self.c]
        self.build()
    
    def build(self):
        E, V, U, W, b, c = self.E, self.V, self.U, self.W, self.b, self.c
        
        x = T.ivector('x')
        y = T.ivector('y')

        # negy is the negative sampling for blackout
        # shape (len(y),k)
        negy = T.imatrix('negy')
        q_w = T.vector('q_w')
        
        def _recurrence(x_t, y_t, neg_y_t, s_t1_prev, q_w):

            # Word embedding layer
            # E hidden word_dim/vocab_dim
            x_e = E[:,x_t]
            
            # GRU Layer 1
            z_t1 = T.nnet.hard_sigmoid(U[0].dot(x_e) + W[0].dot(s_t1_prev) + b[0])
            r_t1 = T.nnet.hard_sigmoid(U[1].dot(x_e) + W[1].dot(s_t1_prev) + b[1])
            c_t1 = T.tanh(U[2].dot(x_e) + W[2].dot(s_t1_prev * r_t1) + b[2])
            s_t1 = (T.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * s_t1_prev

            # probability of output o_t
            # o_t = T.nnet.softmax(V.dot(s_t2) + c)[0]

            # blackout version output probability
            # correct word probability (1,1)
            c_o_t = T.exp(V[y_t].dot(s_t1)+c[y_t])

            # negative word probability (k,1)
            n_o_t = T.exp(V[neg_y_t].dot(s_t1)+c[neg_y_t])

            # sample set probability
            t_o = (q_w[y_t]*c_o_t) + T.sum(q_w[neg_y_t]*n_o_t)

            # positive probability
            c_o_p = q_w[y_t]*c_o_t / t_o

            # negative probability (k,1)
            n_o_p = q_w[neg_y_t]*n_o_t  / t_o

            # cost for each y in blackout
            J_dis = -(T.log(c_o_p) + T.sum(T.log(T.ones_like(n_o_p)-n_o_p)))

            # blackout version discriminative objective function
            return [J_dis, s_t1]
        
        [J, _,_], updates = theano.scan(
            fn=_recurrence,
            sequences=[x,y,negy],
            truncate_gradient=self.bptt_truncate,
            outputs_info=[None,
                          dict(initial=T.zeros(self.hidden_dim))],
            non_sequences=q_w)

        cost = T.sum(J)

        lr=T.scalar("lr")
        gparams=[T.clip(T.grad(cost,p),-10,10) for p in self.params]
        updates=sgd(self.params,gparams,lr)

        self.train=theano.function(inputs=[x,y,negy,q_w,lr],
                                   outputs=cost,
                                   updates=updates)
