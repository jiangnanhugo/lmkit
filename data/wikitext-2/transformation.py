import cPickle as pickle
import numpy as np
from collections import defaultdict

def build_vocab(filename,vocab_size=-1):
    fr=open(filename,'r').read().split('\n')
    word2idx=defaultdict(int)
    word2idx['<s>'] = 0
    word2idx['</s>'] = 0
    for line in fr:
        words=line.strip().split(' ')
        if len(words)==0:
            continue
        word2idx['<s>']+=1
        word2idx['</s>']+=1
        for w in words:
            if len(w)>=1:
                word2idx[w]+=1
    word2idx=sorted(word2idx.items(),cmp=lambda x,y:cmp(x[1],y[1]),reverse=True)

    if vocab_size==-1:
        vocab_size=len(word2idx)
    vocab_freq=word2idx[:vocab_size]


    unk_freq=0
    for item in word2idx[vocab_size:]:
        unk_freq+=item[1]

    if unk_freq>0:
        vocab_freq.append(('<unk>',unk_freq))

    vocab_freq=sorted(vocab_freq,cmp=lambda x,y:cmp(x[1],y[1]),reverse=True)

    word2idx={}
    frequenties=[]
    index=0
    for item in vocab_freq:
        word2idx[item[0]]=index
        index+=1
        frequenties.append(item[1])
    print len(word2idx)
    with open('word2idx.pkl','w')as f:
        pickle.dump(word2idx,f)
    with open('vocab_freq.pkl','w')as f:
        pickle.dump(vocab_freq,f)
    with open('frequenties.pkl','w')as f:
        pickle.dump(frequenties,f)

def corpus2index(filename):
    fr=open(filename,'r').read().split('\n')
    fw=open('idx_'+filename,'w')
    with open('word2idx.pkl','r')as f:
        vocab_dict=pickle.load(f)
    for line in fr:
        words=line.split(' ')
        fw.write(str(vocab_dict['<s>'])+' ')
        for w in words:
            if len(w)>=1:
                if w in vocab_dict:
                    fw.write(str(vocab_dict[w])+' ')
                else:
                    fw.write(str(vocab_dict['<unk>'])+' ')
        fw.write(str(vocab_dict['</s>'])+'\n')
    fw.flush()
    fw.close()





def convert_prefix(cluster_path,word2idx_filepath,mode='word'):
    data=open(cluster_path+'/paths','r').read().split('\n')
    word2idx = pickle.load(open(word2idx_filepath+'/word2idx.pkl', 'r'))
    if mode=='indexes':
        local_word2idx=defaultdict(int)
        for idx in range(len(word2idx)):
            local_word2idx[str(idx)]=idx
        word2idx=local_word2idx

    idx=0
    classes=defaultdict(int)
    class_vocab=defaultdict(list)
    for line in data:
        lined=line.split('\t')
        if len(lined)!=3:
            continue
        binary_prefix,word,_=lined
        if binary_prefix not in classes:
            classes[binary_prefix]=idx
            idx+=1
        #print classes[binary_prefix],word2idx[word]
        class_vocab[classes[binary_prefix]].append(word2idx[word])
    idx2nodes = [[]]*len(word2idx)
    arange_caches=[[]]*len(word2idx)
    gidx=0
    for cid in class_vocab:
        wid=0
        for word in class_vocab[cid]:
            idx2nodes[word]=[cid,wid]
            arange_caches[word]=[gidx,gidx+len(class_vocab[cid])]
            wid+=1
        gidx +=len(class_vocab[cid])


    idx2nodes=np.asarray(idx2nodes, dtype='int32')
    arange_caches=np.asarray(arange_caches,dtype='int32')
    idx2nodes = np.asarray(idx2nodes, dtype='int32')
    with open('prefix.nodes', 'w')as f:
        pickle.dump(idx2nodes, f)

    with open('prefix.arange_caches', 'w')as f:
        pickle.dump(np.asarray(arange_caches, dtype='int32'), f)

    return idx2nodes,arange_caches


suffix='.3000'
build_vocab('wiki.train.tokens'+suffix,vocab_size=-1)

corpus2index('wiki.train.tokens'+suffix)
corpus2index('wiki.valid.tokens'+suffix)
corpus2index('wiki.test.tokens'+suffix)


#convert_prefix('idx_wiki.train-c20-p1.out','.',mode='indexes')
