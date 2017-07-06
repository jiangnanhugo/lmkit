#!/bin/sh
THEANO_FLAGS='floatX=float32,device=cuda0,optimizer=fast_run'   python main.py --train_file ../data/wikitext-103/idx_wiki.train.tokens \
            --valid_file ../data/wikitext-103/idx_wiki.valid.tokens \
            --test_file ../data/wikitext-103/idx_wiki.test.tokens \
            --model_dir ./model/parameters_160078.72.pkl \
            --vocab_size 267736 \
            --batch_size 10 \
            --maxlen 100 \
            --goto_line 0 \
            --valid_freq 4000 --test_freq 8000 --save_freq 30000 
