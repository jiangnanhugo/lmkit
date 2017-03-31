import time

from rnnlm import *
from utils import TextIterator,save_model,load_model

import logging
from argparse import ArgumentParser
import sys
import os
import numpy
numpy.set_printoptions(threshold=numpy.nan)

lr=0.5
p=0
NEPOCH=200

n_input=256
n_hidden=256
cell='gru'
optimizer='sgd'

argument = ArgumentParser(usage='it is usage tip', description='no')
argument.add_argument('--train_file', default='../data/ptb/idx_ptb.train.txt', type=str, help='train dir')
argument.add_argument('--model_dir',default='./model/parameters.pkl',type=str,help='trained model file as checkpoints')
argument.add_argument('--reload_dumps',default=0,type=int,help='reload trained model')
argument.add_argument('--filepath',default='../data/ptb/frequenties.pkl',type=str,help='word frequenties or brown prefix')
argument.add_argument('--valid_file', default='../data/ptb/idx_ptb.valid.txt', type=str, help='valid dir')
argument.add_argument('--test_file', default='../data/ptb/idx_ptb.test.txt', type=str, help='test dir')
argument.add_argument('--vocab_size', default=10001, type=int, help='vocab size')
argument.add_argument('--batch_size', default=5, type=int, help='batch size')
argument.add_argument('--brown_or_huffman', default='huffman', type=str, help='brown or huffman')


args = argument.parse_args()
print args

train_datafile=args.train_file
filepath=args.filepath
valid_datafile=args.valid_file
test_datafile=args.test_file
n_batch=args.batch_size
vocabulary_size=args.vocab_size
brown_or_huffman=args.brown_or_huffman
model_dir=args.model_dir
reload_dumps=args.reload_dumps


n_words_source=-1
disp_freq=20
valid_freq=1000
test_freq=2000
save_freq=20000
clip_freq=9000
pred_freq=20000


def evaluate(test_data,model):
    cost=0
    index=0
    for x,x_mask,(y_node,y_choice,y_bit_mask),y_mask in test_data:
        index+=1
        cost+=model.test(x,x_mask,y_node,y_choice,y_bit_mask,y_mask,x.shape[1])
    return cost/index

def train(lr):
    # Load data
    print 'loading dataset...'

    train_data=TextIterator(train_datafile,filepath,n_words_source=n_words_source,n_batch=n_batch,brown_or_huffman=brown_or_huffman)
    valid_data=TextIterator(valid_datafile,filepath,n_words_source=n_words_source,n_batch=n_batch,brown_or_huffman=brown_or_huffman)
    test_data=TextIterator(test_datafile,filepath,n_words_source=n_words_source,n_batch=n_batch,brown_or_huffman=brown_or_huffman)
    print 'building model...'
    model=RNNLM(n_input,n_hidden,vocabulary_size,cell,optimizer,p)
    if os.path.exists(model_dir) and reload_dumps==1:
        model=load_model(model_dir,model)
    else:
        os.makedirs(os.path.dirname(model_dir))
    print 'training start...'
    start=time.time()
    idx=0
    for epoch in xrange(NEPOCH):
        error=0
        for x,x_mask,(y_node,y_choice,y_bit_mask),y_mask in train_data:
            idx+=1
            cost=model.train(x,x_mask,y_node,y_choice,y_bit_mask,y_mask,x.shape[1],lr)
            error+=cost
            if np.isnan(cost) or np.isinf(cost):
                print 'NaN Or Inf detected!'
                return -1
            if idx % disp_freq==0:
                print 'epoch:',epoch,'idx:',idx,'cost:',error/disp_freq,'ppl:',np.exp(error/disp_freq),'lr:',lr
                error=0
            if idx%save_freq==0:
                print 'dumping...'
                save_model(model_dir,model)
            if idx % valid_freq==0:
                print 'validing....'
                valid_cost=evaluate(valid_data,model)
                print 'valid_cost:',valid_cost,'perplexity:',np.exp(valid_cost)
            if idx % test_freq==0:
                print 'testing...'
                test_cost=evaluate(test_data,model)
                print 'test cost:',test_cost,'perplexity:',np.exp(test_cost)
            if idx%clip_freq==0 and lr >=0.01:
                print 'cliping learning rate:',
                lr=lr*0.9
                print lr
        sys.stdout.flush()

    print "Finished. Time = "+str(time.time()-start)


if __name__ == '__main__':
    train(lr=lr)
