import time
import os
from rnnlm import *
from utils import TextIterator,save_model,calculate_wer,load_model

import logging
from argparse import ArgumentParser 
import sys

lr=0.001
p=0.1
NEPOCH=200

n_input=256
n_hidden=256
cell='gru'

argument = ArgumentParser(usage='it is usage tip', description='no')  
argument.add_argument('--train_file', default='../data/ptb/idx_ptb.train.txt', type=str, help='train dir')  
argument.add_argument('--valid_file', default='../data/ptb/idx_ptb.valid.txt', type=str, help='valid dir')
argument.add_argument('--test_file', default='../data/ptb/idx_ptb.test.txt', type=str, help='test dir')
argument.add_argument('--model_dir', default='./model/parameters_176832.65.pkl', type=str, help='model dir to dump')
argument.add_argument('--goto_line', default=10, type=int, help='goto the specific line index')
argument.add_argument('--vocab_size', default=10001, type=int, help='vocab size')
argument.add_argument('--batch_size', default=10, type=int, help='batch size')
argument.add_argument('--optimizer',default='adam',type=str,help='gradient optimizer: sgd, adam, hf etc.')
argument.add_argument('--mode',default='train',type=str,help='train/valid/test')


args = argument.parse_args()  


train_datafile=args.train_file
valid_datafile=args.valid_file
test_datafile=args.test_file
model_dir=args.model_dir
goto_line=args.goto_line
n_batch=args.batch_size
vocabulary_size=args.vocab_size
optimizer= args.optimizer
n_words_source=-1
disp_freq=10
valid_freq=1000
test_freq=2000
save_freq=20000
clip_freq=9000
pred_freq=20000

def evaluate(test_data,model):
    sumed_cost=[]
    #sumed_wer=[]
    #n_words=[]
    for x,x_mask,y,y_mask in test_data:
        cost,pred_y=model.test(x,x_mask,y,y_mask,x.shape[1])
        #sumed_wer.append(calculate_wer(y,y_mask,np.reshape(pred_y, y.shape)))
        sumed_cost.append(cost)
        #n_words.append(np.sum(y_mask))

    return np.average(sumed_cost)#,np.sum(sumed_wer)/np.sum(n_words)

def train(lr):
    print 'loading dataset...'

    train_data=TextIterator(train_datafile,n_words_source=n_words_source,n_batch=n_batch)
    valid_data=TextIterator(valid_datafile,n_words_source=n_words_source,n_batch=n_batch)
    test_data=TextIterator(test_datafile,n_words_source=n_words_source,n_batch=n_batch)
    print 'building model...'
    model=RNNLM(n_input,n_hidden,vocabulary_size,cell,optimizer,p)
    if os.path.isfile(model_dir):
        model=load_model(model_dir,model)
    if goto_line!=0:
        train_data.goto_line(goto_line)
    print 'training start...'
    start=time.time()
    idx=0
    for epoch in xrange(NEPOCH):
        error=0
        for x,x_mask,y,y_mask in train_data:
            idx+=1
            cost=model.train(x,x_mask,y,y_mask,x.shape[1],lr)
            error+=cost
            if np.isnan(cost) or np.isinf(cost):
                print 'NaN Or Inf detected!'
                return -1
            if idx % disp_freq==0:
                print 'epoch:',epoch,'idx:',idx,'cost:',error/disp_freq,'ppl:',np.exp(error/disp_freq)
                error=0
            if idx%save_freq==0:
                print 'dumping...'
                save_model('./model/parameters_%.2f.pkl'%(time.time()-start),model)
            if idx % valid_freq==0:
                print 'validing....'
                valid_cost=evaluate(valid_data,model)
                print 'valid_cost:',valid_cost,'perplexity:',np.exp(valid_cost)#,"word_error_rate:",mean_wer
            if idx % test_freq==0:
                print 'testing...'
                mean_cost=evaluate(test_data,model)
                print 'test cost:',mean_cost,'perplexity:',np.exp(mean_cost)#,"word_error_rate:",mean_wer
            if idx % pred_freq==0:
                print 'predicting...'
                prediction=model.predict(x,x_mask,x.shape[1])
                print prediction[:100]
            #if idx%clip_freq==0 and lr >=1e-2:
            #    print 'cliping learning rate:',
            #    lr=lr*0.9
            #    print lr
        sys.stdout.flush()

    print "Finished. Time = "+str(time.time()-start)

def test():
    print 'loading dataset...'

    test_data=TextIterator(test_datafile,n_words_source=n_words_source,n_batch=n_batch)
    print 'building model...'
    model=RNNLM(n_input,n_hidden,vocabulary_size,cell,optimizer,p)
    if os.path.isfile(args.model_dir):
        model=load_model(args.model_dir,model)
    print 'testing start...'
    mean_cost,mean_wer=evaluate(test_data,model)
    print 'test cost:',mean_cost,'perplexity:',np.exp(mean_cost),"word_error_rate:",mean_wer


if __name__ == '__main__':
    if args.mode=='train':
        train(lr=lr)
    elif args.mode=='test':
        test()
