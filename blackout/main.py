import time
from utils import *
from grulm import GRULM
from argparse import ArgumentParser

lr=0.001
p=0.1

NEPOCH=100

n_input=256
n_hidden=256
maxlen=100
cell='gru'
argument = ArgumentParser(usage='it is usage tip', description='no')
argument.add_argument('--train_file', default='../data/ptb/idx_ptb.train.txt', type=str, help='train dir')
argument.add_argument('--valid_file', default='../data/ptb/idx_ptb.valid.txt', type=str, help='valid dir')
argument.add_argument('--test_file', default='../data/ptb/idx_ptb.test.txt', type=str, help='test dir')
argument.add_argument('--vocab_size', default=10001, type=int, help='vocab size')
argument.add_argument('--batch_size', default=20, type=int, help='batch size')
argument.add_argument('--vocabfreq_file', default='../data/ptb/vocab_freq.pkl', type=str, help='vocab_freq')
argument.add_argument('--optimizer',default='adam',type=str,help='gradient optimizer: sgd, adam, hf etc.')

args = argument.parse_args()


train_datafile=args.train_file
valid_datafile=args.valid_file
test_datafile=args.test_file
vocabulary_size=args.vocab_size
n_batch=args.batch_size
vocab_freq_file=args.vocabfreq_file
optimizer= args.optimizer
n_words_source=-1


disp_freq=25
sample_freq=200
save_freq=5000
valid_freq=200
test_freq=20
clip_freq=5000


k = vocabulary_size/20
alpha = 0.75

def evaluate(test_data,model):
    cost=0
    index=0
    for (x,y) in test_data:
        predict_error=model.test(x,y)
        index+=predict_error.shape[0]
        #print np.mean(predict_error),np.sum(predict_error),predict_error.shape[0]
        cost+=np.sum(predict_error)
    return cost/index

def train(lr):
    with open(vocab_freq_file,'r') as f:
        vocab_freq=pickle.load(f)
    vocab_p = Q_w(vocab_freq,alpha)

    # Load data
    print 'loading dataset...'
    train_data=TextIterator(train_datafile,n_words_source=n_words_source,maxlen=maxlen)
    valid_data = TextIterator(valid_datafile, n_words_source=n_words_source, maxlen=maxlen)
    test_data=TextIterator(test_datafile,n_words_source=n_words_source,maxlen=maxlen)

    print 'building model...'
    model=GRULM(n_hidden,vocabulary_size,optimizer= optimizer)
    print 'training start...'
    start=time.time()
    for epoch in xrange(NEPOCH):
        error=0
        idx=0
        for (x,y) in train_data:
            idx+=1
            negy=negative_sample(y,k,vocab_p)

            cost=model.train(x, y, negy, vocab_p,lr)
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
            #if idx % pred_freq==0:
                #print 'predicting...'
                #prediction=model.predict(x,x_mask,x.shape[1])
                #print prediction[:100]
            #if idx%clip_freq==0 and lr >=1e-2:
            #    print 'cliping learning rate:',
            #    lr=lr*0.9
            #   print lr

    print "Finished. Time = "+str(time.time()-start)


if __name__ == '__main__':
    train(lr=lr)
