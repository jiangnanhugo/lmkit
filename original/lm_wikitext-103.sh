#!/bin/sh
#THEANO_FLAGS='floatX=float32,device=gpu0,optimizer=fast_run'   
python main.py --train_file ../data/wikitext-103/idx_wiki.train.tokens \
            --valid_file ../data/wikitext-103/idx_wiki.valid.tokens \
            --test_file ../data/wikitext-103/idx_wiki.test.tokens \
            --model_dir ./model/parameters_226887.61.pkl \
            --vocab_size 267736 \
            --batch_size 5 \
            --maxlen 100
            #--goto_line 126000
