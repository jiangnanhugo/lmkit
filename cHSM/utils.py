import numpy as np
import cPickle as pickle


class TextIterator(object):
    def __init__(self,source,prefix_filepath,n_batch,maxlen=None):

        self.source=open(source,'r')
        self.nodes = pickle.load(open(path_file, 'r'))
        self.arange_caches = pickle.load(open(path_file, 'r'))
        self.nodes,self.arange_caches=load_prefix(prefix_filepath)

        self.n_batch=n_batch
        self.maxlen=maxlen
        self.end_of_data=False

    def reconstruct(self,y):
        return self.nodes[y],self.arange_caches[y]

    def __iter__(self):
        return self

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
                s=[int(w) for w in s]
                # filter long sentences
                if self.maxlen>0 and len(s)>self.maxlen:
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
        x, x_mask,y, y_mask=prepare_data(source)
        return x,x_mask,self.reconstruct(y),y_mask

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



def load_prefix(path_file):









if __name__=='__main__':
    load_prefix('idx_wiki.train-c50-p1.out/paths')