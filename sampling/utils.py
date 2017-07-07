import numpy as np
import cPickle as pickle
import theano

def alias_setup(probs):
    K= len(probs)
    q=np.zeros(K)
    J=np.zeros(K,dtype=np.int32)

    # Sort the data into the outcomes with probabilities
    # that are larger and smaller than 1/K;
    smaller=[]
    larger=[]
    for idx,probs in enumerate(probs):
        q[idx]=K*probs
        if q[idx]<1.0:
            smaller.append(idx)
        else:
            larger.append(idx)


    # Loop though and create little binary mixtures that
    # appropriately allocate the laeger outcomes over the
    # overall uniform mixture.
    while len(smaller)>0 and len(larger)>0:
        small=smaller.pop()
        large=larger.pop()

        J[small]=large
        q[large]=q[large]-(1.0-q[small])

        if q[large]<1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J,q


def Q_w(vocab,alpha):
    """
    weight for blackout the 1/relative frequence of the word
    """
    vocab_p = np.ones(len(vocab))
    q_t = 0
    for item in vocab:
        q_t = q_t + float(item[1]**alpha)

    for idx in range(len(vocab)):
        vocab_p[idx] = float(vocab[idx][1]**alpha)/float(q_t)

    return np.asarray(vocab_p,dtype=theano.config.floatX)




def alias_draw(J,q,k,pos_idx):
    '''
    sampling K negative word from q_dis, Sk!=i
    :param J:
    :param q:
    :param k:
    :param pos_idx:
    :return:
    '''
    K=len(J)

    # Draw from the overall uniform mixture.
    kk=np.random.rand(k*3)*K
    kk = [int(np.floor(x)) for x in kk]

    ne_sample=[]
    for it in kk:
        if len(ne_sample)==k:
            break
        if np.random.rand()<q[it]:
            if it !=pos_idx:
                ne_sample.append(it)
        else:
            if J[it]!=pos_idx:
                ne_sample.append(J[it])
    return ne_sample


def negative_sample(pos_y,k,J,q):
    """
    sample for integer vector pos_y
    """

    neg_m = []
    for pos_index in pos_y:
        neg_m.append(alias_draw(J,q,k,pos_index))

    #print neg_m
    return np.asarray(neg_m,dtype=np.int32)


