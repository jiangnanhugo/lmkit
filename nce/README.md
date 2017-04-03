## Noise Contrastive Estimation
Recurrent language model in Theano using noise contrastive estimation (NCE)

Trains a word-level language model on a text file using an GRU. Once the model is trained a piece of text is generated.

Using NCE greatly improves efficiency for word-level language models where the large vocabulary size makes computing softmax inefficient. 

NCE can only be used during the training of the model as it describes a loss function, not a way of outputting predictions. A full softmax is used for generating samples.

Adapted from the character-level model by Eben Olson [here](https://github.com/ebenolson/pydata2015/blob/master/4%20-%20Recurrent%20Networks/RNN%20Character%20Model%20-%202%20Layer.ipynb).
