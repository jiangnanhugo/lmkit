#!/bin/sh

rnn_cell=lstm
model_dir=None
tbptt=5
THEANO_FLAGS="floatX=float32,device=cuda0, optimizer=fast_run"  nohup python main.py --train_file ../data/wikitext-2/idx_wiki.train.tokens \
--valid_file ../data/wikitext-2/idx_wiki.valid.tokens \
--test_file ../data/wikitext-2/idx_wiki.test.tokens \
--vocab_size 33279 \
--batch_size 20 \
--rnn_cell $rnn_cell \
--model_dir $model_dir \
--bptt $tbptt >log/wiki2_tbptt_$tbptt_$rnn_cell.log &


