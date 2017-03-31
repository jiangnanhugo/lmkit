import cPickle as pickle

def build_vocab(filename,vocab_size=-1):
    fr=open(filename,'r').read().split('\n')
    vocab=dict()
    vocab['<s>'] = 0
    vocab['</s>'] = 0
    for line in fr:
        words=line.split(' ')
        vocab['<s>']+=1
        vocab['</s>']+=1
        for w in words:
            if len(w)>=1 and w not in vocab:
                vocab[w]=1
            elif len(w)>=1:
                vocab[w]+=1
    vocab=sorted(vocab.items(),cmp=lambda x,y:cmp(x[1],y[1]),reverse=True)

    if vocab_size==-1:
        vocab_size=len(vocab)
    vocab_freq=vocab[:vocab_size]

    unk_freq=0
    for item in vocab[vocab_size:]:
        unk_freq+=item[1]
    if unk_freq>0:
        vocab_freq.append(('<unk>',unk_freq))

    vocab_freq=sorted(vocab_freq,cmp=lambda x,y:cmp(x[1],y[1]),reverse=True)

    vocab_dict={}
    frequenties=[]
    index=0
    for item in vocab_freq:
        vocab_dict[item[0]]=index
        index+=1
        frequenties.append(item[1])
    print len(vocab_dict)
    with open('vocab_dict.pkl','w')as f:
        pickle.dump(vocab_dict,f)
    with open('vocab_freq.pkl','w')as f:
        pickle.dump(vocab_freq,f)
    with open('frequenties.pkl','w')as f:
        pickle.dump(frequenties,f)

def corpus2index(filename):
    fr=open(filename,'r').read().split('\n')
    fw=open('idx_'+filename,'w')
    with open('vocab_dict.pkl','r')as f:
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

'''
build_vocab('dataset.tr',vocab_size=793471)

corpus2index('dataset.tr')
corpus2index('dataset.te')
'''
build_vocab('./ptb.train.txt')