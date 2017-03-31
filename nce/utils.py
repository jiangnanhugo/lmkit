import numpy as np
import cPickle as pickle
import theano

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

def blackout(vocab_p,k,pos_index):
    """
    sampling K negative word from q_dis, Sk != i
    """
    ne_sample = []
    while len(ne_sample) < k:
        p = np.random.choice(len(vocab_p),1, p=vocab_p)[0]
        if p == pos_index:
            pass
        else:
            ne_sample.append(p)

    return np.asarray(ne_sample)


def negative_sample(pos_y,k,vocab_p):
    """
    blackout sample for integer vector pos_y
    """

    neg_m = []
    for pos_index in pos_y:
        neg_m.append(blackout(vocab_p,k,pos_index))

    return np.asarray(neg_m,dtype=np.int32)



def save_model(f,model):
    ps={}
    for p in model.params:
        ps[p.name]=p.get_value()
    pickle.dump(ps,open(f,'wb'))

def load_model(f,model):
    ps=pickle.load(open(f,'rb'))
    for p in model.params:
        p.set_value(ps[p.name])
    return model

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