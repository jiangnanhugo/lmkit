#!/bin/sh
rnn_cell=fastlstm
valid_freq=4000
test_freq=4000
save_freq=30000
maxlen=-1
batch_size=1

train_file=../data/wikitext-103/idx_wiki.train.tokens
valid_file=../data/wikitext-103/idx_wiki.valid.tokens
test_file=../data/wikitext-103/idx_wiki.test.tokens
vocab_size=267736
check_point=./model/parameters_49337.70.pkl
vocab_freq_file=../data/wikitext-103/vocab_freq.pkl
sampling=blackout
mode=testing
THEANO_FLAGS="floatX=float32,device=cuda2,optimizer=fast_run" python main.py --train_file $train_file \
            --valid_file $valid_file \
            --test_file $test_file \
            --vocab_size $vocab_size \
            --batch_size $batch_size \
            --rnn_cell $rnn_cell \
            --goto_line 0 \
            --maxlen $maxlen \
            --valid_freq $valid_freq \
            --model_dir $check_point\
            --test_freq $test_freq \
            --save_freq $valid_freq \
            --vocab_freq_file $vocab_freq_file \
            --sampling $sampling \
            --mode $mode

