import numpy as np
np.set_printoptions(threshold=np.inf)
import cPickle as pickle
import Queue
import copy
from collections import defaultdict

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
    def __init__(self,source,filepath,n_batch,maxlen=None,n_words_source=-1,brown_or_huffman='huffman',mode='vector'):

        self.source=open(source,'r')
        if brown_or_huffman=='brown':
            print "Brown clusteing"
            self.nodes,self.choices,self.bitmasks=load_brown_prefix(filepath)
        elif brown_or_huffman=='huffman':
            print "Huffman clustering"
            self.nodes, self.choices, self.bitmasks = load_huffman_prefix(filepath)

        self.n_batch=n_batch
        self.maxlen=maxlen
        self.n_words_source=n_words_source
        self.end_of_data=False
        self.mode=mode

    def reconstruct(self,y):
        if self.mode=='matrix':
            nodes = self.nodes[y]
            choices = self.choices[y]
            bitmasks = self.bitmasks[y]

            print 'bitmask',bitmasks.shape
            print 'nonzero',np.count_nonzero(bitmasks, axis=-1).shape
            maxlen = np.max(np.count_nonzero(bitmasks, axis=-1).flatten())
            nodes = nodes[:, :, :maxlen]
            choices = choices[:, :, :maxlen]
            bitmasks = bitmasks[:, :, :maxlen]
            return nodes, choices, bitmasks


        elif self.mode=='vector':
            # rotate the matrix could just solve the problems.
            nodes = self.nodes[y]
            choices = self.choices[y]
            bitmasks = self.bitmasks[y]
            print 'bitmask',bitmasks.shape
            print 'nonzero',np.count_nonzero(bitmasks, axis=-1)
            maxlen = np.max(np.count_nonzero(bitmasks, axis=-1).flatten())
            nodes = nodes[:, :, :maxlen].transpose((2, 0, 1))
            choices = choices[:, :, :maxlen].transpose((2, 0, 1))
            bitmasks = bitmasks[:, :, :maxlen].transpose((2, 0, 1))
            return nodes, choices, bitmasks

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

class Node(object):
    def __init__(self,left=None,right=None,index=None):
        self.left=left
        self.right=right
        self.index=index

    def __repr__(self):
        string=str(self.index)
        if self.left:
            string+=', -1:'+str(self.left.index)
        if self.right:
            string+=', +1:'+str(self.right.index)
        return string

    def preorder(self,polarity=None,param=None,collector=None):
        if collector is None:
            collector=[]
        if polarity is None:
            polarity=[]
        if param is None:
            param=[]
        if self.left:
            if isinstance(self.left[1],Node):
                self.left[1].preorder(polarity+[-1],param+[self.index],collector)
            else:
                collector.append((self.left[1],param+[self.index], polarity + [-1]))
        if self.right:
            if isinstance(self.right[1],Node):
                self.right[1].preorder(polarity+[1],param+[self.index],collector)
            else:
                collector.append((self.right[1],param+[self.index], polarity + [1]))
        return collector


def build_huffman(frequenties):
    Q=Queue.PriorityQueue()
    for v in frequenties:  #((freq,word),index)
        Q.put(v)
    idx=0
    while Q.qsize()>1:
        l,r=Q.get(),Q.get()
        node=Node(l,r,idx)
        idx+=1
        freq=l[0]+r[0]
        Q.put((freq,node))
    return Q.get()[1]


def load_huffman_prefix(freq_file):
    rel_freq=pickle.load(open(freq_file, 'r'))
    freq = zip(rel_freq, range(len(rel_freq)))
    tree = build_huffman(freq)
    x = tree.preorder()
    x=sorted(x, key=lambda z: z[0])

    length_x=[len(it[1]) for it in x]
    vocab_size=len(x)
    maxlen=np.max(length_x)
    nodes=np.zeros((vocab_size,maxlen),dtype='int32')
    choices=np.zeros((vocab_size,maxlen),dtype='float32')
    bit_masks=np.zeros((vocab_size,maxlen),dtype='float32')

    for idx,node,choice in x:
        nodes[idx,:length_x[idx]]=node
        choices[idx,:length_x[idx]]=choice
        bit_masks[idx,:length_x[idx]]=1
    return nodes,choices,bit_masks


def load_brown_prefix(brown_path_file):
    texts=open(brown_path_file,'r')
    word_map=list()
    choice=list()
    node=list()
    maxlen=0
    for line in texts:
        try:
            widx,bitstr,_=line.split()
            choice.append([1 if x == '1' else -1 for x in bitstr])
            node.append([1 if x == '1' else -1 for x in bitstr[:-1]])
            word_map.append(int(widx))
            if len(bitstr)>=maxlen:
                maxlen=len(bitstr)
        except ValueError:
            break
    node=build_brown_node(node,maxlen)

    length_x=[len(it) for it in choice]
    vocab_size=len(choice)
    nodes=np.zeros((vocab_size,maxlen),dtype='int32')
    choices=np.zeros((vocab_size,maxlen),dtype='float32')
    bit_masks=np.zeros((vocab_size,maxlen),dtype='float32')

    for idx in word_map:
        print len(node[idx]),len(choice[idx])
        nodes[idx,:length_x[idx]]=node[idx]
        choices[idx,:length_x[idx]]=choice[idx]
        bit_masks[idx,:length_x[idx]]=1
    return nodes,choices,bit_masks


def build_brown_node(choice,maxlen):
    count = 0
    for col in range(maxlen):
        pre=0
        for row in range(len(choice)):
            if len(choice[row])<=col:
                continue
            if pre==0:
                pre=choice[row][col]+1
            if choice[row][col]!=pre:
                count+=1
                pre=choice[row][col]

            choice[row][col]=count

    node=[[0]+ch[:-1] for ch in choice]
    print "node count",node.shape
    return node



if __name__=='__main__':
    #load_brown_prefix('idx_wiki.train-c50-p1.out/paths')
    '''
    train_data = TextIterator('../data/wikitext-2/idx_wiki.train.tokens', '../data/wikitext-2/frequenties.pkl', n_words_source=-1, n_batch=5,
                              brown_or_huffman='huffman', mode='matrix')
    for x,x_mask,(y_node,y_choice,y_bit_mask),y_mask in train_data:
        print '-'*80
        print y_node.shape
        print '='*80
    '''

    train_data = TextIterator('../data/wikitext-2/idx_wiki.train.tokens', '../data/wikitext-2/wikitext-2_sorted.txt', n_words_source=-1,
                              n_batch=5,
                              brown_or_huffman='brown', mode='matrix')
    for x, x_mask, (y_node, y_choice, y_bit_mask), y_mask in train_data:
        print '-' * 80
        print x.shape
        print y_node.shape
        print '=' * 80

