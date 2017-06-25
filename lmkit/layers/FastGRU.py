import numpy as np
import theano
import theano.tensor as T


class FastGRU(object):
    def __init__(self, rng,
                 n_input, n_hidden,
                 x, E, mask,
                 is_train=1, p=0.5):
        # https://github.com/nyu-dl/dl4mt-tutorial/tree/master/session3
        self.rng = rng

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.f = T.nnet.sigmoid

        self.x = x
        self.E = E
        self.mask = mask
        self.is_train = is_train
        self.p = p

        # Update gate
        init_W = np.asarray(np.random.uniform(low=-np.sqrt(1. / n_input),
                                              high=np.sqrt(1. / n_input),
                                              size=(n_input, n_hidden * 2)),
                            dtype=theano.config.floatX)
        init_U = np.asarray(np.random.uniform(low=-np.sqrt(1. / n_input),
                                              high=np.sqrt(1. / n_input),
                                              size=(n_hidden, n_hidden*2)),
                            dtype=theano.config.floatX)

        init_b = np.zeros((n_hidden * 2), dtype=theano.config.floatX)

        self.W = theano.shared(value=init_W, name='W')
        self.U = theano.shared(value=init_U, name='U')
        self.b = theano.shared(value=init_b, name='b')

        # Cell update
        init_Wx = np.asarray(np.random.uniform(low=-np.sqrt(1. / n_input),
                                               high=np.sqrt(1. / n_input),
                                               size=(n_input, n_hidden)),
                             dtype=theano.config.floatX)
        init_Ux = np.asarray(np.random.uniform(low=-np.sqrt(1. / n_input),
                                               high=np.sqrt(1. / n_input),
                                               size=(n_hidden, n_hidden)),
                             dtype=theano.config.floatX)
        init_bx = np.zeros((n_hidden), dtype=theano.config.floatX)

        self.Wx = theano.shared(value=init_Wx, name='Wx')
        self.Ux = theano.shared(value=init_Ux, name='Ux')
        self.bx = theano.shared(value=init_bx, name='bx')

        # Params
        self.params = [self.W, self.U, self.b, self.Wx, self.Ux, self.bx]

        self.build()

    def build(self):
        state_pre = T.zeros((self.x.shape[-1], self.n_hidden), dtype=theano.config.floatX)
        state_below = T.dot(self.E[self.x,:], self.W) + self.b
        state_belowx = T.dot(self.E[self.x,:], self.Wx) + self.bx

        def split(x, n, dim):
            if x.ndim == 3:
                return x[:, :, n * dim: (n + 1) * dim]
            return x[:, n * dim:(n + 1) * dim]

        def _recurrence(x_t, xx_t, m, h_tm1):
            preact = x_t + T.dot(h_tm1, self.U)

            # reset fate
            r_t = self.f(split(preact, 0, self.n_hidden))
            # Update gate
            z_t = self.f(split(preact, 1, self.n_hidden))

            # Cell update
            c_t = T.tanh(T.dot(h_tm1, self.Ux) * r_t + xx_t)

            # Hidden state
            h_t = (T.ones_like(z_t) - z_t) * c_t + z_t * h_tm1

            # masking
            h_t = h_t * m[:, None]

            return h_t

        h, _ = theano.scan(fn=_recurrence,
                           sequences=[state_below, state_belowx, self.mask],
                           outputs_info=state_pre,
                           truncate_gradient=-1)

        # Dropout
        if self.p > 0:
            drop_mask = self.rng.binomial(n=1, p=1 - self.p, size=h.shape, dtype=theano.config.floatX)
            self.activation = T.switch(self.is_train, h * drop_mask, h * (1 - self.p))
        else:
            self.activation = T.switch(self.is_train, h, h)
