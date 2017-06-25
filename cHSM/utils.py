import numpy as np
import cPickle as pickle
from collections import defaultdict

#np.set_printoptions(threshold=np.nan)
import theano


class TextIterator(object):
    def __init__(self, source, prefix_path, n_batch, maxlen=None):

        self.source = open(source, 'r')
        self.nodes, self.n_class, self.node_maxlen = pickle.load(open(prefix_path , 'r'))

        self.n_batch = n_batch
        self.maxlen = maxlen
        self.end_of_data = False

    def reconstruct(self, y):
        node = self.nodes[y]
        return node.transpose()

    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration
        source = []
        try:
            while True:
                s = self.source.readline()
                if s == "":
                    raise IOError
                s = s.strip().split(' ')
                s = [int(w) for w in s]
                # filter long sentences
                if self.maxlen > 0 and len(s) > self.maxlen:
                    continue
                source.append(s)
                if len(source) >= self.n_batch:
                    break
        except IOError:
            self.end_of_data = True

        if len(source) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration
        x, x_mask, y, y_mask = prepare_data(source)
        return x, x_mask, self.reconstruct(y), y_mask


def prepare_data(seqs_x):
    lengths_x = [len(s) - 1 for s in seqs_x]
    n_samples = len(seqs_x)
    maxlen_x = np.max(lengths_x)

    x = np.zeros((maxlen_x, n_samples)).astype('int32')
    y = np.zeros((maxlen_x, n_samples)).astype('int32')
    x_mask = np.zeros((maxlen_x, n_samples)).astype('float32')
    y_mask = np.zeros((maxlen_x, n_samples)).astype('float32')

    for idx, s_x in enumerate(seqs_x):
        x[:lengths_x[idx], idx] = s_x[:-1]
        y[:lengths_x[idx], idx] = s_x[1:]
        x_mask[:lengths_x[idx], idx] = 1
        y_mask[:lengths_x[idx], idx] = 1

    return x, x_mask, y.flatten(), y_mask.flatten()


# this method was pruned as there not allowed batched slice methods, and add support for dynamic length numpy array.
def convert_prefix_bak(cluster_path, word2idx_filepath, mode='word'):
    data = open(cluster_path + '/paths', 'r').read().split('\n')
    word2idx = pickle.load(open(word2idx_filepath + '/word2idx.pkl', 'r'))
    if mode == 'indexes':
        local_word2idx = defaultdict(int)
        for idx in range(len(word2idx)):
            local_word2idx[str(idx)] = idx
        word2idx = local_word2idx

    idx = 0
    classes = defaultdict(int)
    class_vocab = defaultdict(list)
    for line in data:
        lined = line.split('\t')
        if len(lined) != 3:
            continue
        binary_prefix, word, _ = lined
        if binary_prefix not in classes:
            classes[binary_prefix] = idx
            idx += 1
        # print classes[binary_prefix],word2idx[word]
        class_vocab[classes[binary_prefix]].append(word2idx[word])
    idx2nodes = [[]] * len(word2idx)
    arange_caches = [[]] * len(word2idx)
    gidx = 0
    for cid in class_vocab:
        wid = 0
        for word in class_vocab[cid]:
            idx2nodes[word] = [cid, wid]
            arange_caches[word] = [gidx, gidx + len(class_vocab[cid])]
            wid += 1
        gidx += len(class_vocab[cid])

    idx2nodes = np.asarray(idx2nodes, dtype='int32')
    arange_caches = np.asarray(arange_caches, dtype='int32')
    idx2nodes = np.asarray(idx2nodes, dtype='int32')
    with open('prefix.nodes', 'w')as f:
        pickle.dump(idx2nodes, f)

    with open('prefix.arange_caches', 'w')as f:
        pickle.dump(np.asarray(arange_caches, dtype='int32'), f)

    return idx2nodes, arange_caches


def convert_prefix(cluster_path, word2idx_filepath, mode='word'):
    data = open(cluster_path + '/paths', 'r').read().split('\n')
    word2idx = pickle.load(open(word2idx_filepath + '/word2idx.pkl', 'r'))
    if mode == 'indexes':
        local_word2idx = defaultdict(int)
        for idx in range(len(word2idx)):
            local_word2idx[str(idx)] = idx
        word2idx = local_word2idx

    idx = 0
    classes = defaultdict(int)
    class_vocab = defaultdict(list)
    for line in data:
        lined = line.split('\t')
        if len(lined) != 3:
            continue
        binary_prefix, word, _ = lined
        if binary_prefix not in classes:
            classes[binary_prefix] = idx
            idx += 1
        # print classes[binary_prefix],word2idx[word]
        class_vocab[classes[binary_prefix]].append(word2idx[word])
    idx2nodes = [[]] * len(word2idx)

    node_maxlen = 0
    class_size = len(class_vocab)
    for cid in class_vocab:
        if node_maxlen <= len(class_vocab[cid]):
            node_maxlen = len(class_vocab[cid])
        wid = 0
        for word in class_vocab[cid]:
            idx2nodes[word] = [cid, wid]
            wid += 1

    node_mask = np.zeros((class_size, node_maxlen), dtype=theano.config.floatX)
    for idx in range(len(idx2nodes)):
        x, y = idx2nodes[idx]
        node_mask[x][y] = 1.

    print class_size,node_maxlen
    idx2nodes = np.asarray(idx2nodes, dtype='int32')
    with open('prefix.pkl', 'w')as f:
        pickle.dump((idx2nodes, class_size, node_maxlen), f)

    with open('node_mask.pkl', 'w')as f:
        pickle.dump(node_mask, f)

    return idx2nodes, node_maxlen


if __name__ == '__main__':
    convert_prefix('../data/wikitext-2/idx_wiki.train-c20-p1.out', '../data/wikitext-2', mode='indexes')
