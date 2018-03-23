import time
import os
from rnnlm import *
from lmkit.utils import TextIterator,save_model,calculate_wer,load_model

import logging
from logging.config import fileConfig

fileConfig('../logging_config.ini')
logger = logging.getLogger()

from argparse import ArgumentParser
import itertools
lr=0.001
p=0.1


n_input=512
n_hidden=512


argument = ArgumentParser(usage='it is usage tip', description='no')  
argument.add_argument('--train_file', default='../data/wikitext-2/idx_wiki.train.tokens', type=str, help='train dir')
argument.add_argument('--valid_file', default='../data/wikitext-2/idx_wiki.valid.tokens', type=str, help='valid dir')
argument.add_argument('--test_file', default='../data/wikitext-2/idx_wiki.test.tokens', type=str, help='test dir')
argument.add_argument('--model_dir', default='./model/parameters_123456.pkl', type=str, help='model dir to dump')
argument.add_argument('--goto_line', default=10, type=int, help='goto the specific line index')
argument.add_argument('--vocab_size', default=33279, type=int, help='vocab size')
argument.add_argument('--batch_size', default=10, type=int, help='batch size')
argument.add_argument('--rnn_cell', default='lstm', type=str, help='lstm/gru/fastgru/fastlstm')
argument.add_argument('--optimizer',default='adam',type=str,help='gradient optimizer: sgd, adam, hf etc.')
argument.add_argument('--mode',default='train',type=str,help='train/valid/test')
argument.add_argument('--maxlen',default=256,type=int,help='constrain the maxlen for training')
argument.add_argument('--valid_freq',default=2000,type=int,help='validation frequency')
argument.add_argument('--save_freq',default=2000,type=int,help='save frequency')
argument.add_argument('--test_freq',default=2000,type=int,help='test frequency')
argument.add_argument('--bptt',default=-1,type=int,help='truncated bptt')
argument.add_argument('--nepoch',default=20,type=int,help='running epoch')


args = argument.parse_args()  
NEPOCH=args.nepoch
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

def evaluate_ppl(test_data,model):
    sumed_cost=[]
    n_words=0
    for x,x_mask,y,y_mask in test_data:
        batch_nll,_=model.test(x,x_mask,y,y_mask)
        sumed_cost.append(batch_nll)
        n_words+=np.sum(y_mask)
    return np.sum(sumed_cost)/(1.0*n_words)

def evaluate_wer(test_data,model):
    sumed_wer=[]
    n_words=[]
    for x,x_mask,y,y_mask in test_data:
        _,pred_y=model.test(x,x_mask,y,y_mask)
        sumed_wer.append(calculate_wer(y,y_mask,np.reshape(pred_y, y.shape)))
        n_words.append(np.sum(y_mask))
    return np.sum(sumed_wer)/np.sum(n_words)

def train(lr):
    # Load data
    print 'loading dataset...'

    train_data = TextIterator(train_datafile, n_batch=n_batch, maxlen=maxlen)
    valid_data = TextIterator(valid_datafile, n_batch=n_batch, maxlen=maxlen)
    test_data = TextIterator(test_datafile, n_batch=n_batch, maxlen=maxlen)
    print 'building model...'
    model=RNNLM(n_input,n_hidden,vocabulary_size,rnn_cell,optimizer,p,bptt)
    print 'training start...'
    start=time.time()
    idx=0
    error=[]
    n_words=0
    for epoch in xrange(NEPOCH):
        in_start=time.time()
        for x,x_mask,y,y_mask in train_data:
            idx+=1
            beg_time=time.time()
            #print x.shape
            #print y.shape
            cost,batch_nll=model.train(x,x_mask,y,y_mask,lr)
            error.append(batch_nll)
            n_words+=np.sum(y_mask)
            if np.isnan(cost) or np.isinf(cost):
                print 'NaN Or Inf detected!'
                return -1
            if idx % disp_freq==0:
                error=np.asarray(error).flatten()
                logger.info('epoch: %d idx: %d cost: %f ppl: %f' % (
                    epoch, idx, np.sum(error)/n_words, np.exp(np.sum(error)/n_words)))
                error=[]
                n_words=0
            if idx%save_freq==0:
                filename='./model/param_{}_bptt{}_{:.2f}.pkl'.format(rnn_cell,bptt,(time.time()-start))
                logger.info( 'dumping...'+filename)
                save_model(filename,model)
            if idx % valid_freq==0 :
                logger.info('validing...')
                valid_cost=evaluate_ppl(valid_data,model)
                logger.info('validation cost: %f perplexity: %f' % (valid_cost, np.exp(valid_cost)))
            if idx % test_freq==0 :
                logger.info('testing...')
                test_cost=evaluate_ppl(test_data,model)
                logger.info('test cost: %f perplexity: %f' % (test_cost, np.exp(test_cost)))

  

    print "Finished. Time = "+str(time.time()-start)


def test():
    test_data=TextIterator(test_datafile,n_batch=n_batch)
    valid_data=TextIterator(valid_datafile,n_batch=n_batch)
    model=RNNLM(n_input,n_hidden,vocabulary_size,rnn_cell,optimizer,p)
    if os.path.isfile(args.model_dir):
        print 'loading pretrained model:',args.model_dir
        model=load_model(args.model_dir,model)
    else:
        print args.model_dir,'not found'
    mean_cost=evaluate_ppl(valid_data,model)
    print 'valid perplexity:',np.exp(mean_cost),
    mean_wer=evaluate_wer(valid_data,model)
    print 'valid WER:',mean_wer

    mean_cost=evaluate_ppl(test_data,model)
    print 'test perplexity:',np.exp(mean_cost),
    mean_wer=evaluate_wer(test_data,model)
    print 'test WER:',mean_wer


if __name__ == '__main__':
    if args.mode=='train':
        train(lr=lr)
    elif args.mode=='test':
        test()
