#!/bin/sh
python main.py --train_file ../data/wikitext-2/idx_wiki.train.tokens \
            --valid_file ../data/wikitext-2/idx_wiki.valid.tokens \
            --test_file ../data/wikitext-2/idx_wiki.test.tokens \
            --vocab_size 33279 \
            --batch_size 20
