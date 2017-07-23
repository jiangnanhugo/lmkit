import theano
import theano.tensor as T
from theano import gof,Apply
import  numpy as np


class Softmax(gof.op):
    """
        Softmax activation function
        :math:`\\varphi(\\mathbf{x})_j =
        \\frac{e^{\mathbf{x}_j}}{\sum_{k=1}^K e^{\mathbf{x}_k}}`
        where :math:`K` is the total number of neurons in the layer. This
        activation function gets applied row-wise.

        """
    nin=1
    nout=1
    __props__=()

    def make_node(self,x):
        x=T.as_tensor_variable(x)
        if not x.type.ndim not in (1,2) \
            or x.type.dtype not in T.float_dtypes:
            raise ValueError('x must be 1-d or 2-d tensor of floats Got %s' % x.type)

        if x.ndim==1:
            x.T.shape_padleft(x,n_ones=1)

        return Apply(self,[x],[x.type()])

    def perform(self,node,input_storage,output_storage):
        x,=input_storage
        e_x=np.exp(x - x.max(axis=1)[:,None])
        sm=e_x/e_x.sum(axis=1)[:,None]
        output_storage[0][0]=sm

    def grad(self,inp,grads):
        x,=inp
        g_sm,=grads
        sm=softmax_op(x)
        return [softax_grad(g_sm,sm)]

    def R_op(self,inputs,eval_points):
        # I think the Jacobian is symmetric so the R_op
        # is the same as the grad
        if None in eval_points:
            return [None]
        return self.grad(inputs,eval_points)

    def infer_shape(self,nonde,shape):
        return shape

    def c_headers(self):
        return ['<iostream>','cmath']

    @staticmethod
    def c_code_template(dtype):
