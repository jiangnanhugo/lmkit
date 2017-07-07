#!/bin/sh
num_operations=3000
train_file=../../data/wikitext-2/wiki.train.tokens
codes_file=../../data/wikitext-2/wiki.train.tokens.codes
valid_file=../../data/wikitext-2/wiki.valid.tokens
test_file=../../data/wikitext-2/wiki.test.tokens
python learn_bpe.py -s $num_operations <$train_file >$codes_file
python apply_bpe.py -c $codes_file <$train_file > $train_file.$num_operations
python apply_bpe.py -c $codes_file <$valid_file >$valid_file.$num_operations
python apply_bpe.py -c $codes_file <$test_file >$test_file.$num_operations
