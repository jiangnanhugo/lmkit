#!/bin/sh
THEANO_FLAGS='floatX=float32,device=gpu2,optimizer=fast_run'   python main.py --train_file ../data/wikitext-103/idx_wiki.train.tokens \
            --valid_file ../data/wikitext-103/idx_wiki.valid.tokens \
            --test_file ../data/wikitext-103/idx_wiki.test.tokens \
            --vocab_size 267736 \
            --batch_size 2
