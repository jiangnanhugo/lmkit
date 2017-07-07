#!/bin/sh
rnn_cell=rnnblock.lstm

train_file=../data/obw/idx_train.txt
valid_file=../data/obw/idx_valid.txt
test_file=../data/obw/idx_test.txt

valid_freq=50000
test_freq=50000
batch_size=6
maxlen=60

check_point=None #./model/parameters_3732.93.pkl

THEANO_FLAGS="floatX=float32,device=cuda3,mode=FAST_RUN,lib.cnmem=1"  python main.py --train_file $train_file \
            --valid_file $valid_file \
            --test_file $test_file \
            --vocab_size 793472 \
            --batch_size $batch_size \
            --rnn_cell $rnn_cell \
            --goto_line 0 \
            --valid_freq $valid_freq \
            --test_freq $test_freq  \
            --model_dir $check_point\
            --maxlen $maxlen
            
