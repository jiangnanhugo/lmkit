#!/bin/sh
THEANO_FLAGS="floatX=float32,device=gpu1,lib.cnmem=0.3"  python main.py --train_file ../data/wikitext-2/idx_wiki.train.tokens \
            --valid_file ../data/wikitext-2/idx_wiki.valid.tokens \
            --test_file ../data/wikitext-2/idx_wiki.test.tokens \
            --vocab_size 33279 \
            --batch_size 20 
