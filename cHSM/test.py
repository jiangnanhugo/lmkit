import theano
import theano.tensor as T
import numpy as np

rng=np.random.RandomState(12345)
maxlen=30
batch_size=10
class_size=108
max_word_dim=109
hidden_size=256

init_wp_matrix=np.asarray(rng.uniform(low=-1, high=1,size=(class_size,max_word_dim,hidden_size)),dtype=theano.config.floatX)
wp_matrix=theano.shared(value=init_wp_matrix,name="word_V",borrow=True)

init_wb=np.zeros((class_size,max_word_dim),dtype=theano.config.floatX)
wb=theano.shared(value=init_wb,name='word_bias',borrow=True)


y_node=T.imatrix('y_nodes')
x=T.fmatrix('hidden layer output')



wps=wp_matrix[y_node[0],:,:]
wbias=wb[y_node[0],:]
z=T.nnet.logsoftmax(T.sum(wps*x.dimshuffle(0,'x',1),axis=-1)+wbias)
output=z.take(y_node[1])

f=theano.function(inputs=[x,y_node],outputs=output)
fw=theano.function(inputs=[y_node],outputs=wbias)
fw2=theano.function(inputs=[y_node],outputs=wps)

np_y_node=rng.randint(low=0,high=class_size,size=(2,batch_size))
np_x=rng.rand(batch_size,hidden_size).astype('float32')


print fw(np_y_node).shape

print fw2(np_y_node).shape
print f(np_x,np_y_node).shape
print f(np_x,np_y_node)