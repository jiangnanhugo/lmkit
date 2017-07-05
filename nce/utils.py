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




class TextIterator:
    def __init__(self,source,maxlen,n_words_source=-1):

        self.source=open(source,'r')
        #self.word2index=word2index
        self.maxlen=maxlen
        self.n_words_source=n_words_source
        self.end_of_data=False

    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data=False
            self.reset()
            raise StopIteration
        try:
            while True:
                s=self.source.readline()
                if s=="":
                    self.end_of_data=False
                    self.reset()
                    raise StopIteration
                s=s.strip().split(' ')

                s=[int(w) for w in s]
                if self.n_words_source>0:
                    s=[int(w) if int(w) <self.n_words_source else 3 for w in s]
                # filter long sentences
                if len(s)>self.maxlen:
                    continue
                return (np.asarray(s[:-1],dtype=np.int32),np.asarray(s[1:],dtype=np.int32))

        except IOError:
            self.end_of_data=True

def prepare_data(seqs_x):
    lengths_x=[len(s)-1 for s in seqs_x]
    n_samples=len(seqs_x)
    maxlen_x=np.max(lengths_x)

    x=np.zeros((maxlen_x,n_samples)).astype('int32')
    y=np.zeros((maxlen_x,n_samples)).astype('int32')

    for idx,s_x in enumerate(seqs_x):
        x[:lengths_x[idx],idx]=s_x[:-1]
        y[:lengths_x[idx],idx]=s_x[1:]

    return x,y