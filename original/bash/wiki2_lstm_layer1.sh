#!/bin/sh
rnn_cell=rnnblock.lstm
valid_freq=1000
test_freq=2000
save_freq=2000
batch_size=20

train_file=../data/wikitext-2/idx_wiki.train.tokens
valid_file=../data/wikitext-2/idx_wiki.valid.tokens
test_file=../data/wikitext-2/idx_wiki.test.tokens
vocab_size=33279

# mode=train
# check_point=None #

mode=test
check_point=./model/param_rnnblock.lstm_bptt-1_2744.44.pkl
batch_size=1
THEANO_FLAGS="floatX=float32,device=cuda1,mode=FAST_RUN" nohup python main.py --train_file $train_file \
            --valid_file $valid_file \
            --test_file $test_file \
            --vocab_size $vocab_size \
            --batch_size $batch_size \
            --rnn_cell $rnn_cell \
            --goto_line 0 \
            --valid_freq $valid_freq \
            --model_dir $check_point\
            --test_freq $test_freq \
            --save_freq $save_freq \
            --mode $mode \
            >> log/wiki2_lstm_layer1.log &
