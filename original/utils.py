import numpy as np
import cPickle as pickle

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

class TextIterator(object):
    def __init__(self,source,n_batch,maxlen=None,n_words_source=-1):

        self.source=open(source,'r')
        self.n_batch=n_batch
        self.maxlen=maxlen
        self.n_words_source=n_words_source
        self.end_of_data=False

    def __iter__(self):
        return self

    def goto_line(self, line_index):
        for _ in range(line_index):
            self.source.readline()

    def reset(self):
        self.source.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data=False
            self.reset()
            raise StopIteration
        source=[]
        try:
            while True:
                s=self.source.readline()
                if s=="":
                    raise IOError
                s=s.strip().split(' ')

                if self.n_words_source>0:
                    s=[int(w) if int(w) <self.n_words_source else 3 for w in s]
                # filter long sentences
                if self.maxlen and len(s)>self.maxlen:
                    continue
                source.append(s)
                if len(source)>=self.n_batch:
                    break
        except IOError:
            self.end_of_data=True

        if len(source)<=0:
            self.end_of_data=False
            self.reset()
            raise StopIteration
        return prepare_data(source)

def prepare_data(seqs_x):
    lengths_x=[len(s)-1 for s in seqs_x]
    n_samples=len(seqs_x)
    maxlen_x=np.max(lengths_x)

    x=np.zeros((maxlen_x,n_samples)).astype('int32')
    y=np.zeros((maxlen_x,n_samples)).astype('int32')
    x_mask=np.zeros((maxlen_x,n_samples)).astype('float32')
    y_mask=np.zeros((maxlen_x,n_samples)).astype('float32')

    for idx,s_x in enumerate(seqs_x):
        x[:lengths_x[idx],idx]=s_x[:-1]
        y[:lengths_x[idx],idx]=s_x[1:]
        x_mask[:lengths_x[idx],idx]=1
        y_mask[:lengths_x[idx],idx]=1

    return x,x_mask,y,y_mask


def wer(r, h):
    """
    Calculation of WER with Levenshtein distance.
    Works only for iterables up to 254 elements (uint8).
    O(nm) time ans space complexity.
    Parameters
    ----------
    """
    # initialisation

    d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.int32)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion    = d[i][j-1] + 1
                deletion     = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)]


def prune_sentence(x,xmask):
    sent=[]
    for it_x,it_mask in zip(x,xmask):
        if it_mask==1.:
          sent.append(it_x)
        else: break
    return sent


def calculate_wer(y,y_mask,pred_y):
    maxlen,batch_size=y.shape
    wer_score=0
    for b in range(batch_size):
        sent_y=prune_sentence(y[:,b],y_mask[:,b])
        sent_pred=prune_sentence(pred_y[:,b],y_mask[:,b])

        wer_score+=wer(sent_y,sent_pred)
        #print wer(sent_y,sent_pred),np.sum(y_mask[:b])
    return wer_score

if __name__ == "__main__":
    print wer([197, 2249, 185, 12, 0, 8241, 25, 13, 4, 9, 197, 136, 297, 47, 221, 20, 13, 4, 3],
              [1, 500, 1, 1, 6440, 1584, 1, 1, 9980, 5116, 5613, 500, 500, 9475, 9475, 73, 7, 9976, 8938])
    print([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,])
    print wer("who is there".split(), "is there".split())
    print wer("who is there".split(), "".split())