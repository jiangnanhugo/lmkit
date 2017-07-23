import time
from utils import *
from rnnlm import RNNLM
import cPickle as pickle
from argparse import ArgumentParser
from lmkit.utils import *

import logging
from logging.config import fileConfig
fileConfig('../logging_config.ini')
logger = logging.getLogger()

lr=0.06
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
argument.add_argument('--bptt',default=-1,type=int,help='truncated bptt')
argument.add_argument('--epoch',default=10,type=int,help='maximum epochs')
argument.add_argument('--sampling',choices=["blackout",'nce'],help='Model type: either nce or blakcout', default="nce")


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



k = vocabulary_size/200
alpha = 0.75

def evaluate(test_data,model,mode='no'):
    sumed_cost=0
    sumed_wer=[]
    n_words=[]
    idx=0
    for x,x_mask,y,y_mask in test_data:
        nll,pred_y=model.test(x,x_mask,y,y_mask)
        if mode=='wer':
            sumed_wer.append(calculate_wer(y,y_mask,np.reshape(pred_y, y.shape)))
            n_words.append(np.sum(y_mask))
        else:
            sumed_wer.append(1.)
            n_words.append(1.)
        sumed_cost+=nll
        idx+=np.sum(y_mask)
        #
    return sumed_cost/(1.0*idx),np.sum(sumed_wer)/np.sum(n_words)


def train(lr):
    with open(vocab_freq_file,'r') as f:
        vocab_freq=pickle.load(f)
    vocab_p = Q_w(vocab_freq,alpha)
    J,q=alias_setup(vocab_p)

    # Load data
    print 'loading dataset...'
    train_data=TextIterator(train_datafile,n_batch=n_batch,maxlen=maxlen)
    valid_data = TextIterator(valid_datafile,n_batch=n_batch,maxlen=maxlen)
    test_data=TextIterator(test_datafile,n_batch=n_batch,maxlen=maxlen)

    print 'building model...'
    model=RNNLM(n_input,n_hidden,vocabulary_size, cell=rnn_cell,optimizer=optimizer,p=p,q_w=vocab_p,k=k)
    if os.path.isfile(model_dir):
        print 'loading checkpoint parameters....',model_dir
        model=load_model(model_dir,model)
    if goto_line>0:
        train_data.goto_line(goto_line)
        print 'goto line:',goto_line
    print 'training start...'
    start=time.time()

    idx = 0
    for epoch in xrange(NEPOCH):
        error = 0
        for x,x_mask,y,y_mask in train_data:
            idx+=1
            negy=negative_sample(y,y_mask,k,J,q)
            cost=model.train(x,x_mask, y, negy,y_mask,lr)
            #print cost
            error+=cost
            if np.isnan(cost) or np.isinf(cost):
                print 'NaN Or Inf detected!'
                return -1
            if idx % disp_freq==0:
                logger.info('epoch: %d idx: %d cost: %f ppl: %f' % (
                    epoch, idx, (error / disp_freq), np.exp(error / (1.0 * disp_freq))))
                error=0
            if idx%save_freq==0:
                logger.info( 'dumping...')
                save_model('./model/parameters_%.2f.pkl'%(time.time()-start),model)
            if idx % valid_freq==0 :
                logger.info('validing...')
                valid_cost,wer=evaluate(valid_data,model)
                logger.info('validation cost: %f perplexity: %f,word_error_rate:%f' % (valid_cost, np.exp(valid_cost), wer))
            if idx % test_freq==0 :
                logger.info('testing...')
                test_cost,wer=evaluate(test_data,model)
                logger.info('test cost: %f perplexity: %f,word_error_rate:%f' % (test_cost, np.exp(test_cost),wer))

    print "Finished. Time = "+str(time.time()-start)

def test():
    with open(vocab_freq_file,'r') as f:
        vocab_freq=pickle.load(f)
    vocab_p = Q_w(vocab_freq,alpha)
    J,q=alias_setup(vocab_p)
    valid_data = TextIterator(valid_datafile,n_batch=n_batch,maxlen=maxlen)
    test_data=TextIterator(test_datafile,n_batch=n_batch,maxlen=maxlen)
    model=RNNLM(n_input,n_hidden,vocabulary_size, cell=rnn_cell,optimizer=optimizer,p=p,q_w=vocab_p,k=k)
    if os.path.isfile(args.model_dir):
        print 'loading pretrained model:',args.model_dir
        model=load_model(args.model_dir,model)
    else:
        print args.model_dir,'not found'
    valid_cost, wer = evaluate(valid_data, model,'wer')
    logger.info('validation cost: %f perplexity: %f,word_error_rate:%f' % (valid_cost, np.exp(valid_cost), wer))
    test_cost, wer = evaluate(test_data, model,'wer')
    logger.info('test cost: %f perplexity: %f,word_error_rate:%f' % (test_cost, np.exp(test_cost), wer))


if __name__ == '__main__':
    if args.mode=='train':
        train(lr=lr)
    elif args.mode=='testing':
        test()
