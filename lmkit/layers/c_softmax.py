import numpy as np
import theano
import theano.tensor as T
import logging


class level_softmax(object):

    def __init__(self,shape,x,y_node,y_adj,maskY):
        self.rng=np.random.RandomState(12345)
        self.prefix="level_softmax_"
        self.in_size,self.out_size=shape
        self.x=x
        self.y_node=y_node
        self.y_adj=y_adj
        self.maskY=maskY

        wp_val=np.asarray(self.rng.uniform(low=-np.sqrt(6./(self.in_size)),
                                           high=np.sqrt(6./(self.in_size+2)),
                                           size=(self.out_size,self.in_size)),dtype=theano.config.floatX)
        self.wp_matrix=theano.shared(value=wp_val,name="V_soft",borrow=True)

        self.params=[self.wp_matrix,]
        self.build_graph()

    def build_graph(self):
        wps=self.wp_matrix[self.y_adj[0] : self.y_adj[-1]]
        node=T.sum(wps * self.x.dimshuffle(0,1,'x',2),axis=-1)
        probs=T.nnet.logsoftmax(node)

        self.activation=T.sum(probs.take(self.y_node),axis=-1)