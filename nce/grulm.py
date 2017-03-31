import numpy as np
import theano as theano
from updates import sgd
import theano.tensor as T

class GRULM:
    def __init__(self,hidden_dim, word_dim, bptt_truncate=-1,k=10):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.k=k
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

        # negy is the negative sampling for nce
        # shape (len(y),k)
        negy = T.lmatrix('negy')
        q_w = T.vector('q_w')
        
        def forward_prop_step(x_t, y_t, neg_y_t, s_t1_prev, q_w):

            # Word embedding layer
            # E hidden word_dim/vocab_dim
            x_e = E[:,x_t]
            
            # GRU Layer 1
            z_t1 = T.nnet.sigmoid(U[0].dot(x_e) + W[0].dot(s_t1_prev) + b[0])
            r_t1 = T.nnet.sigmoid(U[1].dot(x_e) + W[1].dot(s_t1_prev) + b[1])
            c_t1 = T.tanh(U[2].dot(x_e) + W[2].dot(s_t1_prev * r_t1) + b[2])
            s_t1 = (T.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * s_t1_prev
            
            
            # Final output calculation
            # Theano's softmax returns a matrix with one row, we only need the row


            # probability of output o_t
            # o_t = T.nnet.softmax(V.dot(s_t2) + c)[0]

            # noise contrastive estimation version output probability

            # correct word probability (1,1)
            c_o_t = T.exp(V[y_t].dot(s_t1)+c[y_t])

            # negative word probability (k,1)
            n_o_t = T.exp(V[neg_y_t].dot(s_t1)+c[neg_y_t])

            # positive probability
            c_o_p = c_o_t / (c_o_t+self.k*q_w[y_t])

            # negative probability (k,1)
            n_o_p = q_w[neg_y_t]/(n_o_t+self.k*q_w[neg_y_t])

            # cost for each y in nce
            cost = -(T.log(c_o_p) + T.sum(T.log(n_o_p)))

            return [cost, s_t1]
        
        [j, _, _], updates = theano.scan(
            forward_prop_step,
            sequences=[x,y,negy],
            truncate_gradient=self.bptt_truncate,
            outputs_info=[None,
                          dict(initial=T.zeros(self.hidden_dim))],
            non_sequences=q_w)
        
        # prediction = T.argmax(o, axis=1)
        cost = T.sum(j)
        
        lr=T.scalar("lr")
        gparams=[T.clip(T.grad(cost,p),-10,10) for p in self.params]
        updates=sgd(self.params,gparams,lr)

        self.train=theano.function(inputs=[x,y,negy,q_w,lr],
                                   outputs=cost,
                                   updates=updates)
