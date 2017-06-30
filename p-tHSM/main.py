import time

from rnnlm import RNNLM

from utils import TextIterator
from lmkit.utils import save_model,load_model

import logging
from logging.config import fileConfig
fileConfig('../logging_config.ini')
logger=logging.getLogger()
from argparse import ArgumentParser
import sys
import os
import numpy as np
np.set_printoptions(threshold=np.nan)

lr=0.001
p=0.1
NEPOCH=6

n_input=256
n_hidden=256
cell='gru'
optimizer='adam'

argument = ArgumentParser(usage='it is usage tip', description='no')
argument.add_argument('--train_file', default='../data/ptb/idx_ptb.train.txt', type=str, help='train dir')
argument.add_argument('--model_dir',default='./model/parameters.pkl',type=str,help='trained model file as checkpoints')
argument.add_argument('--reload_dumps',default=0,type=int,help='reload trained model')
argument.add_argument('--filepath',default='../data/ptb/frequenties.pkl',type=str,help='word frequenties or brown prefix')
argument.add_argument('--word2idx_path',default='../data/wikitext-2/word2idx.pkl',type=str,help='word to idx pickle')
argument.add_argument('--valid_file', default='../data/ptb/idx_ptb.valid.txt', type=str, help='valid dir')
argument.add_argument('--test_file', default='../data/ptb/idx_ptb.test.txt', type=str, help='test dir')
argument.add_argument('--vocab_size', default=10001, type=int, help='vocab size')
argument.add_argument('--batch_size', default=5, type=int, help='batch size')
argument.add_argument('--brown_or_huffman', default='huffman', type=str, help='brown or huffman')
argument.add_argument('--matrix_or_vector',default='vector',type=str,help='use matrix or vector to build hierarchical softmax')
argument.add_argument('--mode',default='train',type=str,help='train/valid/test')


args = argument.parse_args()
print args

train_datafile=args.train_file
word2idx_path=args.word2idx_path
filepath=args.filepath
valid_datafile=args.valid_file
test_datafile=args.test_file
n_batch=args.batch_size
vocabulary_size=args.vocab_size
brown_or_huffman=args.brown_or_huffman
matrix_or_vector=args.matrix_or_vector
model_dir=args.model_dir
mode=args.mode
reload_dumps=args.reload_dumps


disp_freq=20
valid_freq=1000
test_freq=1000
save_freq=5000
#clip_freq=9000
#pred_freq=20000

def evaluate(test_data,model):
    nll=0
    index=0
    for x,x_mask,(y_node,y_choice,y_bit_mask),y_mask in test_data:
        index+=np.sum(y_mask)
        nll+=model.test(x,x_mask,y_node,y_choice,y_bit_mask,y_mask)
    return nll/(index*1.)

def train(lr):
    # Load data
    logger.info('loading dataset...')

    train_data=TextIterator(train_datafile,filepath,n_batch=n_batch,brown_or_huffman=brown_or_huffman,mode=matrix_or_vector,word2idx_path=word2idx_path)
    valid_data=TextIterator(valid_datafile,filepath,n_batch=n_batch,brown_or_huffman=brown_or_huffman,mode=matrix_or_vector,word2idx_path=word2idx_path)
    test_data=TextIterator(test_datafile,filepath,n_batch=n_batch,brown_or_huffman=brown_or_huffman,mode=matrix_or_vector,word2idx_path=word2idx_path)
    logger.info('building model...')
    model=RNNLM(n_input,n_hidden,vocabulary_size,cell,optimizer,p=p,mode=matrix_or_vector)
    if os.path.exists(model_dir) and reload_dumps==1:
        logger.info( 'loading parameters from: %s'% model_dir)
        model=load_model(model_dir,model)
    else:
        logger.info("init parameters....")
    logger.info( 'training start...')
    start=time.time()
    idx=0
    for epoch in xrange(NEPOCH):
        error=0
        for x,x_mask,(y_node,y_choice,y_bit_mask),y_mask in train_data:
            idx+=1
            cost=model.train(x,x_mask,y_node,y_choice,y_bit_mask,y_mask,lr)
            error+=cost
            if np.isnan(cost) or np.isinf(cost):
                print 'NaN Or Inf detected!'
                return -1
            if idx % disp_freq==0:
                logger.info( 'epoch: %d idx: %d cost: %f ppl: %f'%(epoch,idx,error/disp_freq,np.exp(error/(1.0*disp_freq))))#,'lr:',lr
                error=0
            if idx%save_freq==0:
                logger.info('dumping...')
                save_model('./model/parameters_%.2f.pkl'%(time.time()-start),model)
            if idx % valid_freq==0:
                logger.info( 'validing....')
                valid_cost=evaluate(valid_data,model)
                logger.info('valid_cost: %f perplexity: %f'%(valid_cost,np.exp(valid_cost)))
            if idx % test_freq==0:
                logger.info('testing...')
                test_cost=evaluate(test_data,model)
                logger.info('test cost: %f perplexity: %f' %(test_cost,np.exp(test_cost)))
            #if idx%clip_freq==0 and lr >=0.01:
            #    print 'cliping learning rate:',
            #    lr=lr*0.9
            #    print lr
        sys.stdout.flush()

    print "Finished. Time = "+str(time.time()-start)


def test():
    valid_data=TextIterator(valid_datafile,filepath,n_batch=n_batch,brown_or_huffman=brown_or_huffman,mode=matrix_or_vector,word2idx_path=word2idx_path)
    test_data=TextIterator(test_datafile,filepath,n_batch=n_batch,brown_or_huffman=brown_or_huffman,mode=matrix_or_vector,word2idx_path=word2idx_path)
    model=RNNLM(n_input,n_hidden,vocabulary_size,cell,optimizer,p,mode=matrix_or_vector)
    if os.path.isfile(args.model_dir):
        print 'loading pretrained model:',args.model_dir
        model=load_model(args.model_dir,model)
    else:
        print args.model_dir,'not found'
    mean_cost=evaluate(valid_data,model)
    print 'valid cost:',mean_cost,'perplexity:',np.exp(mean_cost)#,"word_error_rate:",mean_wer
    mean_cost=evaluate(test_data,model)
    print 'test cost:',mean_cost,'perplexity:',np.exp(mean_cost)


if __name__ == '__main__':
    if args.mode=='train':
        train(lr=lr)
    elif args.mode=='testing':
        test()
