#!/bin/sh
#THEANO_FLAGS="floatX=float32,device=cuda3,mode=FAST_RUN,lib.cnmem=0.9"  
rnn_cell=fastgru
model_dir=None
python main.py --train_file ../data/wikitext-2/idx_wiki.train.tokens \
            --valid_file ../data/wikitext-2/idx_wiki.valid.tokens \
            --test_file ../data/wikitext-2/idx_wiki.test.tokens \
            --vocab_size 33279 \
            --batch_size 20 \
            --rnn_cell $rnn_cell \
            --model_dir $model_dir
