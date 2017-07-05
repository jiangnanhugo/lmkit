import time
from utils import *
from grulm import GRULM
import cPickle as pickle
from argparse import ArgumentParser
from lmkit.utils import *

lr=0.001
p=0.1
NEPOCH=100

n_input=256   # embedding of input word
n_hidden=256  # hidden state layer size
argument = ArgumentParser(usage='it is usage tip', description='no')
argument.add_argument('--train_file', default='../data/wikitext-2/idx_wiki.train.tokens', type=str, help='train dir')
argument.add_argument('--valid_file', default='../data/wikitext-2/idx_wiki.valid.tokens', type=str, help='valid dir')
argument.add_argument('--test_file', default='../data/wikitext-2/idx_wiki.test.tokens', type=str, help='test dir')
argument.add_argument('--goto_line', default=10, type=int, help='goto the specific line index')
argument.add_argument('--model_dir', default='./model/parameters_123456.pkl', type=str, help='model dir to dump')
argument.add_argument('--vocab_size', default=33279, type=int, help='vocab size')
argument.add_argument('--batch_size', default=10, type=int, help='batch size')
argument.add_argument('--rnn_cell', default='lstm', type=str, help='lstm/gru/fastgru/fastlstm')
argument.add_argument('--optimizer',default='adam',type=str,help='gradient optimizer: sgd, adam, hf etc.')
argument.add_argument('--mode',default='train',type=str,help='train/valid/test')
argument.add_argument('--maxlen',default=256,type=int,help='constrain the maxlen for training')
argument.add_argument('--vocab_freq_file', default='../data/wikitext-2/vocab_freq.pkl', type=str, help='vocab_freq')
argument.add_argument('--valid_freq',default=2000,type=int,help='validation frequency')
argument.add_argument('--save_freq',default=20000,type=int,help='save frequency')
argument.add_argument('--test_freq',default=2000,type=int,help='test frequency')

args = argument.parse_args()


train_datafile=args.train_file
valid_datafile=args.valid_file
test_datafile=args.test_file
model_dir=args.model_dir
goto_line=args.goto_line
n_batch=args.batch_size
vocabulary_size=args.vocab_size
rnn_cell=args.rnn_cell
optimizer= args.optimizer
bptt=args.bptt
maxlen=args.maxlen
disp_freq=50
valid_freq=args.valid_freq
test_freq=args.test_freq
save_freq=args.save_freq
vocab_freq_file=args.vocab_freq_file
n_words_source=-1



k = vocabulary_size/20
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
    train_data=TextIterator(train_datafile,maxlen=maxlen)
    valid_data = TextIterator(valid_datafile,maxlen=maxlen)
    test_data=TextIterator(test_datafile,maxlen=maxlen)

    print 'building model...'
    model=GRULM(n_hidden,vocabulary_size,vocab_p,k,optimizer=optimizer)
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
