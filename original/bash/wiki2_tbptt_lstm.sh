#!/bin/sh

rnn_cell=lstm
model_dir=None
# tbptt=3
# THEANO_FLAGS="floatX=float32,device=cuda0, optimizer=fast_run"  nohup python main.py --train_file ../data/wikitext-2/idx_wiki.train.tokens \
# --valid_file ../data/wikitext-2/idx_wiki.valid.tokens \
# --test_file ../data/wikitext-2/idx_wiki.test.tokens \
# --vocab_size 33279 \
# --batch_size 20 \
# --rnn_cell $rnn_cell \
# --model_dir $model_dir \
# --bptt $tbptt >log/wiki2_tbptt_${tbptt}_$rnn_cell.log &


# tbptt=4
# THEANO_FLAGS="floatX=float32,device=cuda1, optimizer=fast_run"  nohup python main.py --train_file ../data/wikitext-2/idx_wiki.train.tokens \
# --valid_file ../data/wikitext-2/idx_wiki.valid.tokens \
# --test_file ../data/wikitext-2/idx_wiki.test.tokens \
# --vocab_size 33279 \
# --batch_size 20 \
# --rnn_cell $rnn_cell \
# --model_dir $model_dir \
# --bptt $tbptt >log/wiki2_tbptt_${tbptt}_$rnn_cell.log &

# tbptt=5
# THEANO_FLAGS="floatX=float32,device=cuda2, optimizer=fast_run"  nohup python main.py --train_file ../data/wikitext-2/idx_wiki.train.tokens \
# --valid_file ../data/wikitext-2/idx_wiki.valid.tokens \
# --test_file ../data/wikitext-2/idx_wiki.test.tokens \
# --vocab_size 33279 \
# --batch_size 20 \
# --rnn_cell $rnn_cell \
# --model_dir $model_dir \
# --bptt $tbptt >log/wiki2_tbptt_${tbptt}_$rnn_cell.log &
# tbptt=6
# THEANO_FLAGS="floatX=float32,device=cuda3, optimizer=fast_run"  nohup python main.py --train_file ../data/wikitext-2/idx_wiki.train.tokens \
# --valid_file ../data/wikitext-2/idx_wiki.valid.tokens \
# --test_file ../data/wikitext-2/idx_wiki.test.tokens \
# --vocab_size 33279 \
# --batch_size 20 \
# --rnn_cell $rnn_cell \
# --model_dir $model_dir \
# --bptt $tbptt >log/wiki2_tbptt_${tbptt}_$rnn_cell.log &


tbptt=7
THEANO_FLAGS="floatX=float32,device=cuda0, optimizer=fast_run"  nohup python main.py --train_file ../data/wikitext-2/idx_wiki.train.tokens \
--valid_file ../data/wikitext-2/idx_wiki.valid.tokens \
--test_file ../data/wikitext-2/idx_wiki.test.tokens \
--vocab_size 33279 \
--batch_size 20 \
--rnn_cell $rnn_cell \
--model_dir $model_dir \
--bptt $tbptt >log/wiki2_tbptt_${tbptt}_$rnn_cell.log &
tbptt=8
THEANO_FLAGS="floatX=float32,device=cuda1, optimizer=fast_run"  nohup python main.py --train_file ../data/wikitext-2/idx_wiki.train.tokens \
--valid_file ../data/wikitext-2/idx_wiki.valid.tokens \
--test_file ../data/wikitext-2/idx_wiki.test.tokens \
--vocab_size 33279 \
--batch_size 20 \
--rnn_cell $rnn_cell \
--model_dir $model_dir \
--bptt $tbptt >log/wiki2_tbptt_${tbptt}_$rnn_cell.log &
tbptt=9
THEANO_FLAGS="floatX=float32,device=cuda2, optimizer=fast_run"  nohup python main.py --train_file ../data/wikitext-2/idx_wiki.train.tokens \
--valid_file ../data/wikitext-2/idx_wiki.valid.tokens \
--test_file ../data/wikitext-2/idx_wiki.test.tokens \
--vocab_size 33279 \
--batch_size 20 \
--rnn_cell $rnn_cell \
--model_dir $model_dir \
--bptt $tbptt >log/wiki2_tbptt_${tbptt}_$rnn_cell.log &
tbptt=10
THEANO_FLAGS="floatX=float32,device=cuda3, optimizer=fast_run"  nohup python main.py --train_file ../data/wikitext-2/idx_wiki.train.tokens \
--valid_file ../data/wikitext-2/idx_wiki.valid.tokens \
--test_file ../data/wikitext-2/idx_wiki.test.tokens \
--vocab_size 33279 \
--batch_size 20 \
--rnn_cell $rnn_cell \
--model_dir $model_dir \
--bptt $tbptt >log/wiki2_tbptt_${tbptt}_$rnn_cell.log &



# tbptt=11
# THEANO_FLAGS="floatX=float32,device=cuda0, optimizer=fast_run"  nohup python main.py --train_file ../data/wikitext-2/idx_wiki.train.tokens \
# --valid_file ../data/wikitext-2/idx_wiki.valid.tokens \
# --test_file ../data/wikitext-2/idx_wiki.test.tokens \
# --vocab_size 33279 \
# --batch_size 20 \
# --rnn_cell $rnn_cell \
# --model_dir $model_dir \
# --bptt $tbptt >log/wiki2_tbptt_${tbptt}_$rnn_cell.log &


# tbptt=12
# THEANO_FLAGS="floatX=float32,device=cuda1, optimizer=fast_run"  nohup python main.py --train_file ../data/wikitext-2/idx_wiki.train.tokens \
# --valid_file ../data/wikitext-2/idx_wiki.valid.tokens \
# --test_file ../data/wikitext-2/idx_wiki.test.tokens \
# --vocab_size 33279 \
# --batch_size 20 \
# --rnn_cell $rnn_cell \
# --model_dir $model_dir \
# --bptt $tbptt >log/wiki2_tbptt_${tbptt}_$rnn_cell.log &

# tbptt=13
# THEANO_FLAGS="floatX=float32,device=cuda2, optimizer=fast_run"  nohup python main.py --train_file ../data/wikitext-2/idx_wiki.train.tokens \
# --valid_file ../data/wikitext-2/idx_wiki.valid.tokens \
# --test_file ../data/wikitext-2/idx_wiki.test.tokens \
# --vocab_size 33279 \
# --batch_size 20 \
# --rnn_cell $rnn_cell \
# --model_dir $model_dir \
# --bptt $tbptt >log/wiki2_tbptt_${tbptt}_$rnn_cell.log &
# tbptt=14
# THEANO_FLAGS="floatX=float32,device=cuda3, optimizer=fast_run"  nohup python main.py --train_file ../data/wikitext-2/idx_wiki.train.tokens \
# --valid_file ../data/wikitext-2/idx_wiki.valid.tokens \
# --test_file ../data/wikitext-2/idx_wiki.test.tokens \
# --vocab_size 33279 \
# --batch_size 20 \
# --rnn_cell $rnn_cell \
# --model_dir $model_dir \
# --bptt $tbptt >log/wiki2_tbptt_${tbptt}_$rnn_cell.log &