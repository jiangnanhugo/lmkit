import time
import os
from rnnlm import *
from lmkit.utils import TextIterator,save_model,calculate_wer,load_model

import logging
from argparse import ArgumentParser
import sys

lr=0.001
p=0.1
NEPOCH=1
n_input=256
n_hidden=256


argument = ArgumentParser(usage='it is usage tip', description='no')  
argument.add_argument('--train_file', default='../data/ptb/idx_ptb.train.txt', type=str, help='train dir')  
argument.add_argument('--valid_file', default='../data/ptb/idx_ptb.valid.txt', type=str, help='valid dir')
argument.add_argument('--test_file', default='../data/ptb/idx_ptb.test.txt', type=str, help='test dir')
argument.add_argument('--model_dir', default='./model/parameters_176832.65.pkl', type=str, help='model dir to dump')
argument.add_argument('--goto_line', default=10, type=int, help='goto the specific line index')
argument.add_argument('--vocab_size', default=10001, type=int, help='vocab size')
argument.add_argument('--batch_size', default=10, type=int, help='batch size')
argument.add_argument('--rnn_cell', default='fastlstm', type=str, help='lstm/gru/fastgru/fastlstm')
argument.add_argument('--optimizer',default='adam',type=str,help='gradient optimizer: sgd, adam, hf etc.')
argument.add_argument('--mode',default='train',type=str,help='train/valid/test')
argument.add_argument('--maxlen',default=256,type=int,help='constrain the maxlen for training')
argument.add_argument('--valid_freq',default=2000,type=int,help='validation frequency')
argument.add_argument('--save_freq',default=20000,type=int,help='save frequency')
argument.add_argument('--test_freq',default=2000,type=int,help='test frequency')
argument.add_argument('--bptt',default=-1,type=int,help='truncated bptt')




args = argument.parse_args()  
print args

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

def evaluate(test_data,model):
    sumed_cost=0
    #sumed_wer=[]
    #n_words=[]
    idx=0
    for x,x_mask,y,y_mask in test_data:
        nll=model.test(x,x_mask,y,y_mask)
        #sumed_wer.append(calculate_wer(y,y_mask,np.reshape(pred_y, y.shape)))
        sumed_cost+=nll
        idx+=np.sum(y_mask)
        #n_words.append(np.sum(y_mask))
    return sumed_cost/(1.0*idx)#,np.sum(sumed_wer)/np.sum(n_words)

def train(lr):
    print 'loading dataset...'
    train_data=TextIterator(train_datafile,n_batch=n_batch,maxlen=maxlen)
    valid_data=TextIterator(valid_datafile,n_batch=n_batch,maxlen=maxlen)
    test_data=TextIterator(test_datafile,n_batch=n_batch,maxlen=maxlen)
    print 'building model...'
    model=RNNLM(n_input,n_hidden,vocabulary_size,rnn_cell,optimizer,p,bptt)
    if os.path.isfile(model_dir):
        print 'loading checkpoint parameters....',model_dir
        model=load_model(model_dir,model)
    if goto_line!=0:
        train_data.goto_line(goto_line)
        print 'goto line:',goto_line
    print 'training start...'
    start=time.time()
    idx=goto_line
    for epoch in xrange(NEPOCH):
        error=0
        for x,x_mask,y,y_mask in train_data:
            idx+=1
            cost=model.train(x,x_mask,y,y_mask,lr)
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
            #if idx % pred_freq==0:
            #    print 'predicting...'
            #    prediction=model.predict(x,x_mask,x.shape[1])
            #    print prediction[:100]
            #if idx%clip_freq==0 and lr >=1e-2:
            #    print 'cliping learning rate:',
            #    lr=lr*0.9
            #    print lr
        sys.stdout.flush()
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
    mean_cost=evaluate(valid_data,model)
    print 'valid cost:',mean_cost,'perplexity:',np.exp(mean_cost)#,"word_error_rate:",mean_wer
    mean_cost=evaluate(test_data,model)
    print 'test cost:',mean_cost,'perplexity:',np.exp(mean_cost)


if __name__ == '__main__':
    if args.mode=='train':
        train(lr=lr)
    elif args.mode=='testing':
        test()
