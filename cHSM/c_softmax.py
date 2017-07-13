import numpy as np
import theano
import theano.tensor as T
import logging



def log_softmax_masked(x,mask):
    rebased_x=x-x.max(axis=1,keepdims=True)
    e_x=rebased_x*mask-T.log(T.sum(mask*T.exp(rebased_x),axis=1,keepdims=True))

    return e_x/e_x.sum(axis=1,keepdims=True)


# class-based hierarchical softmax
class C_softmax(object):

    def __init__(self,shape,x,y_node,y_node_mask,node_maxlen):
        """
        :param shape: used for passing dimension parameters.
        :param x: [maxlen*batch_size,hidden_size], output of last layer, input of this layer.
        :param y_node: [2, maxlen* batch_size], 2 stands for (class_id,word_id)
        :param y_arange_cache:  denotes the start range to the end range for the parameters tensors.
        :param maskY: used for masking out not real word's
        """
        self.rng=np.random.RandomState(12345)
        # class_size: class size
        # out_size: vocabulary size
        self.in_size,self.class_size,self.out_size=shape
        self.x=x.reshape([-1, x.shape[-1]])
        self.y_node=y_node
        self.node_maxlen=node_maxlen


        init_cp_matrix = np.asarray(self.rng.uniform(low=-np.sqrt(6. / (self.in_size)),
                                                   high=np.sqrt(6. / (self.in_size)),
                                                   size=(self.in_size, self.class_size)), dtype=theano.config.floatX)
        self.cp_matrix = theano.shared(value=init_cp_matrix, name="class_V", borrow=True)
        # bias for class layer
        init_cb = np.zeros((self.class_size), dtype=theano.config.floatX)
        self.cb = theano.shared(value=init_cb, name='class_bias', borrow=True)

        init_wp_matrix=np.asarray(self.rng.uniform(low=-np.sqrt(6./(self.in_size)),
                                                high=np.sqrt(6./(self.in_size)),
                                                size=(self.class_size,self.node_maxlen,self.in_size)),dtype=theano.config.floatX)
        self.wp_matrix=theano.shared(value=init_wp_matrix,name="word_V",borrow=True)


        self.y_node_mask = theano.shared(value=y_node_mask, name="y_node_mask", borrow=True)

        # bias for word layer
        init_wb=np.zeros((self.class_size,self.node_maxlen),dtype=theano.config.floatX)
        self.wb=theano.shared(value=init_wb,name='word_bias',borrow=True)

        self.params=[self.cp_matrix,self.cb,self.wp_matrix,self.wb]
        self.build_graph()

    def build_graph(self):

        '''
        self.x: [maxlen* batch_size, hidden_size]
        cp_matrix: hidden_size, class_size
        cbias: class_size
        T.dot(self.x,self.cp_matrix): [maxlen* batch_size, class_size]
        log_class_probs: [maxlen* batch_size, class_size]
        '''
        log_class_probs=T.nnet.logsoftmax(T.dot(self.x,self.cp_matrix)+self.cb)


        '''
        self.wb: vocabulary_size
        wp_matrix: [class_size, max_word_dim, hidden_size]
        self.y_node[0]: (1,bath_size*maxlen)
        wps: [batch_size*maxlen,max_word_dim,hidden_size]
        wbias: [batch_size*maxlen,max_word_dim]
        '''
        wps = self.wp_matrix[self.y_node[0],:,:]
        wbias=self.wb[self.y_node[0],:]

        """
        self.x.dimshuffle(0,'x',1): [maxlen*batch_size,1, hidden_size]
        wps: [batch_size*maxlen,max_word_dim,hidden_size]
        node: [maxlen*batch_size,max_word_dim]
        """
        node=T.sum(wps* self.x.dimshuffle(0,'x',1) ,axis=-1)+wbias

        """
        mask: [maxlen* batch_size, max_word_dim]
        """
        m=self.y_node_mask[self.y_node[0],:,:]
        """
        log_word_probs: [maxlen* batch_size, max_word_dim], normalized probability for words inside each class.
        """
        #log_word_probs=log_softmax_masked(node,m)
        log_word_probs = T.nnet.logsoftmax(node* m)


        # word_log_softmax  +  class_log_softmax
        logprobs=log_class_probs.take(self.y_node[0])+log_word_probs.take(self.y_node[1])

        # [maxlen* batch_size, 1]
        # last dimension denotes the log probability of this word.
        self.predicted=log_softmax_masked(node,m)
        self.activation=logprobs
        self.word_prob=log_word_probs.take(self.y_node[1])
        self.node=node

        # temp
        #temp=T.nnet.softmax(T.dot(self.x,self.cp_matrix)+self.cb)
        #self.log_class_probs=-T.nnet.categorical_crossentropy(temp,self.y_node[0])
        #self.log_class_probs2=log_class_probs.take(self.y_node[0])

