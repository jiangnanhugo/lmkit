import numpy as np
import theano
import theano.tensor as T

def sgd(params,gparams, lr=0.01):
    return [(p,p-lr*gp)for p,gp in zip(params,gparams)]


def adam(params,gparams,learning_rate=0.01,beta1=0.9,beta2=0.999,epsilon=1e-8):
    """
     Adam: A Method for Stochastic Optimization.
        https://arxiv.org/pdf/1412.6980.pdf
    :param cost:
    :param params:
    :param learning_rate:
    :param beta1:
    :param beta2:
    :param epsilon:
    :return:
    """
    updates=[]
    t_tm1=theano.shared(value=np.float32(0))
    t=t_tm1+1
    lr_t=learning_rate * T.sqrt( 1 - beta2**t)/(1- beta1**t)

    for p,g in zip(params,gparams):
        value=p.get_value(borrow=True)
        m= theano.shared(np.zeros(value.shape,dtype=value.dtype))
        v=theano.shared(np.zeros(value.shape,dtype=value.dtype))

        m_t = beta1 * m + (1 - beta1) * g   ## momentum
        v_t = beta2 * v + (1 - beta2) * g**2   ## adagrad

        g_t= m_t/ (T.sqrt(v_t) + epsilon)
        p_t = p - (lr_t *g_t)


        updates.append((m,m_t))
        updates.append((v,v_t))
        updates.append((p,p_t))
    updates.append((t,t+1))

    return updates