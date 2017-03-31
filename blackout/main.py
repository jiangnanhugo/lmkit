import time
from utils import *
from grulm import GRULM
from argparse import ArgumentParser

lr=0.01
p=0

NEPOCH=100

n_input=20
n_hidden=20
maxlen=100
cell='gru'
optimizer='sgd'
argument = ArgumentParser(usage='it is usage tip', description='no')
argument.add_argument('--train_file', default='../data/ptb/idx_ptb.train.txt', type=str, help='train dir')
argument.add_argument('--valid_file', default='../data/ptb/idx_ptb.valid.txt', type=str, help='valid dir')
argument.add_argument('--test_file', default='../data/ptb/idx_ptb.test.txt', type=str, help='test dir')
argument.add_argument('--vocab_size', default=10001, type=int, help='vocab size')
argument.add_argument('--batch_size', default=10, type=int, help='batch size')
argument.add_argument('--vocab_freq_file', default='../data/ptb/vocab_freq.pkl', type=str, help='vocab_freq')

args = argument.parse_args()


train_datafile=args.train_file
valid_datafile=args.valid_file
test_datafile=args.test_file
vocabulary_size=args.vocab_size
n_batch=args.batch_size
vocab_freq_file=args.vocab_freq_file
n_words_source=-1


disp_freq=100
sample_freq=200
save_freq=5000


k = 200
alpha = 0.75


def train():
    with open(vocab_freq_file,'r') as f:
        vocab_freq=pickle.load(f)
    vocab_p = Q_w(vocab_freq,alpha)

    # Load data
    print 'loading dataset...'
    train_data=TextIterator(train_datafile,n_words_source=n_words_source,maxlen=maxlen)
    test_data=TextIterator(test_datafile,n_words_source=n_words_source,maxlen=maxlen)

    print 'building model...'
    model=GRULM(n_hidden,vocabulary_size)
    print 'training start...'
    start=time.time()
    for epoch in xrange(NEPOCH):
        error=0
        idx=0
        in_start=time.time()
        for (x,y) in train_data:
            if x.shape[1]!=n_batch:
                continue
            idx+=1
            negy=negative_sample(y,k,vocab_p)

            cost=model.train(x, y, negy, vocab_p,lr)
            print 'index:',idx,'cost:',cost
            error+=np.sum(cost)
            if np.isnan(cost) or np.isinf(cost):
                print 'NaN Or Inf detected!'
                return -1
            if idx % disp_freq==0:
                print 'epoch:',epoch,'idx:',idx,'cost:',error/disp_freq
                error=0
            if idx%save_freq==0:
                print 'dumping...'
                save_model('model/parameters_%.2f.pkl'%(time.time()-start),model)
            if idx % sample_freq==0:
                print 'Sampling....'
                #y_pred=model.predict(x,x_mask,n_batch)
                #print y_pred

    print "Finished. Time = "+str(time.time()-start)


if __name__ == '__main__':
    train()
