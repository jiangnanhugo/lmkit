import time
from lmkit.utils import save_model,load_model
from utils import *
from grulm import GRULM
import cPickle as pickle
from argparse import ArgumentParser

lr=0.01
p=0.5
NEPOCH=100

n_input=30
n_hidden=20
maxlen=30
cell='gru'
optimizer='adam'
argument = ArgumentParser(usage='it is usage tip', description='no')
argument.add_argument('--train_file', default='../data/wikitext-2/idx_wiki.train.tokens', type=str, help='train dir')
argument.add_argument('--valid_file', default='../data/wikitext-2/idx_wiki.valid.tokens', type=str, help='valid dir')
argument.add_argument('--test_file', default='../data/wikitext-2/idx_wiki.test.tokens', type=str, help='test dir')
argument.add_argument('--vocab_size', default=33279, type=int, help='vocab size')
argument.add_argument('--batch_size', default=10, type=int, help='batch size')
argument.add_argument('--vocab_freq_file', default='../data/wikitext-2/vocab_freq.pkl', type=str, help='vocab_freq')

args = argument.parse_args()


train_datafile=args.train_file
valid_datafile=args.valid_file
test_datafile=args.test_file
vocabulary_size=args.vocab_size
n_batch=args.batch_size
vocab_freq_file=args.vocab_freq_file
n_words_source=-1

disp_freq=100
sample_freq=200000
save_freq=500000
valid_freq=100000
test_freq=2000000
clip_freq=5000000

k = 200
alpha = 0.75

def evaluate(test_data,model):
    sumed_cost=0
    index=0
    for (x,y) in test_data:
        index+=1
        sumed_cost+=model.test(x,y)
    return sumed_cost/index


def train(lr):
    with open(vocab_freq_file,'r') as f:
        vocab_freq=pickle.load(f)
    vocab_p = Q_w(vocab_freq,alpha)
    J,q=alias_setup(vocab_p)

    # Load data
    print 'loading dataset...'
    train_data=TextIterator(train_datafile,n_words_source=n_words_source,maxlen=maxlen)
    valid_data = TextIterator(valid_datafile, n_words_source=n_words_source, maxlen=maxlen)
    test_data=TextIterator(test_datafile,n_words_source=n_words_source,maxlen=maxlen)

    print 'building model...'
    model=GRULM(n_hidden,vocabulary_size,vocab_p,k)
    print 'training start...'
    start=time.time()
    for epoch in xrange(NEPOCH):
        error=0
        idx=0
        for (x,y) in train_data:
            idx+=1
            negy=negative_sample(y,k,J,q)
            cost=model.train(x, y, negy,lr)
            error+=cost
            if np.isnan(cost) or np.isinf(cost):
                print 'NaN Or Inf detected!'
                return -1
            if idx % disp_freq==0:
                print 'epoch:',epoch,'idx:',idx,'cost:',error/disp_freq
                error=0
            if idx%save_freq==0:
                print 'dumping...'
                save_model('model/parameters_%.2f.pkl'%(time.time()-start),model)
            if idx % valid_freq == 0:
                print 'valding...'
                valid_cost = evaluate(valid_data, model)
                print 'valid cost:', valid_cost, 'perplexity:', np.exp(valid_cost)
            if idx % test_freq==0:
                print 'testing...'
                test_cost=evaluate(test_data,model)
                print 'test cost:',test_cost,'perplexity:',np.exp(test_cost)


    print "Finished. Time = "+str(time.time()-start)


if __name__ == '__main__':
    train(lr=lr)
