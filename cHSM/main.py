import time

from rnnlm import *
from utils import TextIterator
from lmkit.utils import load_model, save_model

import logging
from logging.config import fileConfig

fileConfig('../logging_config.ini')
np.set_printoptions(threshold=np.nan)
logger = logging.getLogger()
from argparse import ArgumentParser
import sys
import os

argument = ArgumentParser(usage='it is usage tip', description='no')
argument.add_argument('--n_input', default=256, type=int, help='word vector dimension')
argument.add_argument('--n_hidden', default=256, type=int, help='hidden vector dimension')
argument.add_argument('--nepoch', default=6, type=int, help='epoch over the dataset')
argument.add_argument('--optimizer', default='adam', type=str, help='gradient optimizer')
argument.add_argument('--droput_drop', default=0.1, type=float, help='dropout rate of dropped')
argument.add_argument('--lr', default=0.001, type=float, help='learning rate')
argument.add_argument('--model_dir', default='./model/parameters.pkl', type=str,
                      help='trained model file as checkpoints')
argument.add_argument('--reload_dumps', default=0, type=int, help='reload trained model')
argument.add_argument('--train_file', default='../data/wikitext-2/idx_wiki.train.tokens', type=str, help='train dir')
argument.add_argument('--valid_file', default='../data/wikitext-2/idx_wiki.valid.tokens', type=str, help='valid dir')
argument.add_argument('--test_file', default='../data/wikitext-2/idx_wiki.test.tokens', type=str, help='test dir')
argument.add_argument('--node_mask_path', default='node_mask.pkl', type=str, help='node mask pickle file')
argument.add_argument('--prefix_path', default='prefix.pkl', type=str,
                      help='classes and words prefix configures pickle file.')

argument.add_argument('--valid_freq', default=2000, type=int, help='validation frequency')
argument.add_argument('--save_freq', default=20000, type=int, help='save frequency')
argument.add_argument('--test_freq', default=2000, type=int, help='test frequency')
argument.add_argument('--goto_line', default=0, type=int, help='goto the specific line index')

argument.add_argument('--vocab_size', default=33287, type=int, help='vocab size')
argument.add_argument('--batch_size', default=2, type=int, help='batch size')
argument.add_argument('--rnn_cell', default='gru', type=str, help='recurrent unit type')
argument.add_argument('--mode', default='train', type=str, help='train/valid/test')
argument.add_argument('--maxlen', default=200, type=int, help='constrain the maxlen for training')

args = argument.parse_args()

train_datafile = args.train_file
valid_datafile = args.valid_file
test_datafile = args.test_file
n_batch = args.batch_size
vocab_size = args.vocab_size
rnn_cell = args.rnn_cell
prefix_path = args.prefix_path
node_mask_path = args.node_mask_path
maxlen = args.maxlen
model_dir = args.model_dir
reload_dumps = args.reload_dumps

disp_freq = 4
goto_line = args.goto_line
valid_freq = args.valid_freq
test_freq = args.test_freq
save_freq = args.save_freq
n_input=args.n_input
n_hidden=args.n_hidden
lr=args.lr
optimizer=args.optimizer
NEPOCH=args.nepoch
p=args.dropout_drop

def evaluate(test_data, model):
    sumed_cost = 0
    sumed_wer = []
    n_words = []
    idx = 0
    for x, x_mask, y, y_mask in test_data:
        # nll,pred_y=model.test(x,x_mask,y,y_mask)
        # sumed_wer.append(calculate_wer(y,y_mask,np.reshape(pred_y, y.shape)))
        sumed_wer.append(1.)
        sumed_cost += 1.0
        idx += 1  # np.sum(y_mask)
        # n_words.append(np.sum(y_mask))
        n_words.append(1.)
    return sumed_cost / (1.0 * idx), np.sum(sumed_wer) / np.sum(n_words)


def train_by_batch(lr):
    # Load data
    logger.info('loading dataset...')

    train_data = TextIterator(train_datafile, prefix_path=prefix_path, n_batch=n_batch, maxlen=maxlen)
    valid_data = TextIterator(valid_datafile, prefix_path=prefix_path, n_batch=n_batch, maxlen=maxlen)
    test_data = TextIterator(test_datafile, prefix_path=prefix_path, n_batch=n_batch, maxlen=maxlen)

    logger.info('building model...')
    model = RNNLM(n_input, n_hidden, vocab_size, n_class=train_data.n_class, node_len=train_data.node_maxlen,
                  rnn_cell=rnn_cell, optimizer=optimizer, p=p,node_mask_path=node_mask_path)

    if os.path.exists(model_dir) and reload_dumps == 1:
        logger.info('loading parameters from: %s' % model_dir)
        model = load_model(model_dir, model)
    else:
        print "init parameters...."
    if goto_line > 0:
        train_data.goto_line(goto_line)
        print 'goto line:', goto_line
    print 'training start...'
    start = time.time()
    idx = goto_line
    logger.info('training start...')
    for epoch in xrange(NEPOCH):
        error = 0
        for x, x_mask, y_node, y_mask in train_data:
            idx += 1
            #cost, logprob = model.train(x, x_mask, y_node, y_mask, lr)
            cost = model.train(x, x_mask, y_node, y_mask, lr)
            error += cost
            if np.isnan(cost) or np.isinf(cost):
                print 'NaN Or Inf detected!'
                return -1
            if idx % disp_freq == 0:
                logger.info('epoch: %d idx: %d cost: %f ppl: %f' % (
                    epoch, idx, error / disp_freq, np.exp(error / (1.0 * disp_freq))))
                error = 0
            if idx % save_freq == 0:
                print 'dumping...'
                save_model('./model/parameters_%.2f.pkl' % (time.time() - start), model)
            if idx % valid_freq == 0:
                logger.info('validing....')
                valid_cost = evaluate(valid_data, model)
                logger.info('valid_cost: %f perplexity: %f' % (valid_cost, np.exp(valid_cost)))
            if idx % test_freq == 0:
                logger.info('testing...')
                test_cost = evaluate(test_data, model)
                logger.info('test cost: %f perplexity: %f' % (test_cost, np.exp(test_cost)))


        sys.stdout.flush()

    print "Finished. Time = " + str(time.time() - start)


def train_by_epoch(lr):
    # Load data
    logger.info('loading dataset...')

    train_data = TextIterator(train_datafile, prefix_path=prefix_path, n_batch=n_batch, maxlen=maxlen)
    valid_data = TextIterator(valid_datafile, prefix_path=prefix_path, n_batch=n_batch, maxlen=maxlen)
    test_data = TextIterator(test_datafile, prefix_path=prefix_path, n_batch=n_batch, maxlen=maxlen)

    logger.info('building model...')
    model = RNNLM(n_input, n_hidden, vocab_size, n_class=train_data.n_class, node_len=train_data.node_maxlen,
                  rnn_cell=rnn_cell, optimizer=optimizer, p=p,node_mask_path=node_mask_path)

    if os.path.exists(model_dir) and reload_dumps == 1:
        logger.info('loading parameters from: %s' % model_dir)
        model = load_model(model_dir, model)
    else:
        print "init parameters...."
    if goto_line > 0:
        train_data.goto_line(goto_line)
        print 'goto line:', goto_line

    start = time.time()
    idx = goto_line
    logger.info('training start...')
    for epoch in xrange(NEPOCH):
        error = 0
        for x, x_mask, y_node, y_mask in train_data:
            idx += 1
            #cost, logprob = model.train(x, x_mask, y_node, y_mask, lr)
            cost = model.train(x, x_mask, y_node, y_mask, lr)
            error += cost
            if np.isnan(cost) or np.isinf(cost):
                print 'NaN Or Inf detected!'
                return -1
            if idx % disp_freq == 0:
                logger.info('epoch: %d idx: %d cost: %f ppl: %f' % (
                    epoch, idx, error / disp_freq, np.exp(error / (1.0 * disp_freq))))
                error = 0
        save_model('./model/parameters_%.2f.pkl' % (time.time() - start), model)
        valid_cost = evaluate(valid_data, model)
        logger.info('valid_cost: %f perplexity: %f' % (valid_cost, np.exp(valid_cost)))
        test_cost = evaluate(test_data, model)
        logger.info('test cost: %f perplexity: %f' % (test_cost, np.exp(test_cost)))
        sys.stdout.flush()

    print "Finished. Time = " + str(time.time() - start)


def test():
    valid_data = TextIterator(valid_datafile, prefix_path=prefix_path, n_batch=n_batch)
    test_data = TextIterator(test_datafile, prefix_path=prefix_path, n_batch=n_batch)
    model = RNNLM(n_input, n_hidden, vocab_size, rnn_cell=rnn_cell, optimizer=optimizer, p=p,
                  n_class=valid_data.n_class, node_len=valid_data.node_maxlen, node_mask_path=node_mask_path)
    if os.path.isfile(args.model_dir):
        print 'loading pretrained model:', args.model_dir
        model = load_model(args.model_dir, model)
    else:
        print args.model_dir, 'not found'
    mean_cost = evaluate(valid_data, model)
    print 'valid cost:', mean_cost, 'perplexity:', np.exp(mean_cost)  # ,"word_error_rate:",mean_wer
    mean_cost = evaluate(test_data, model)
    print 'test cost:', mean_cost, 'perplexity:', np.exp(mean_cost)


if __name__ == '__main__':
    if args.mode == 'train':
        train_by_epoch(lr=lr)
    elif args.mode == 'testing':
        test()
