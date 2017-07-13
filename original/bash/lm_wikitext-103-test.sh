#!/bin/sh
rnn_cell=rnnblock.lstm
check_point=./model/parameters_3732.93.pkl
valid_freq=1000
test_freq=2000
batch_size=20

train_file=../data/wikitext-103/idx_wiki.valid.tokens
valid_file=../data/wikitext-103/idx_wiki.test.tokens
test_file=../data/wikitext-2/idx_wiki.test.tokens
vocab_size=33279
mode=testing
THEANO_FLAGS="floatX=float32,device=cuda3,mode=FAST_RUN" python main.py --train_file $train_file \
            --valid_file $valid_file \
            --test_file $test_file \
            --vocab_size $vocab_size \
            --batch_size $batch_size \
            --rnn_cell $rnn_cell \
            --goto_line 0 \
            --valid_freq $valid_freq \
            --model_dir $check_point\
            --test_freq $test_freq \
            --mode $mode


## 58915.87
# valid cost: 6.32173968008 perplexity: 556.540352924
# test cost: 6.32761876862 perplexity: 559.821939855

# loading pretrained model: ./model/parameters_25681.67.pkl
# valid cost: 6.35649898782 perplexity: 576.225448532
# test cost: 6.33455561984 perplexity: 563.718841847

#loading pretrained model: ./model/parameters_34184.57.pkl
#valid cost: 6.29029866341 perplexity: 539.314378505
#test cost: 6.27793062292 perplexity: 532.68519594

#loading pretrained model: ./model/parameters_43054.13.pkl
#valid cost: 6.2461580489 perplexity: 516.026462923
#test cost: 6.24466952515 perplexity: 515.258916672

#loading pretrained model: ./model/parameters_51824.11.pkl
#valid cost: 6.23813183081 perplexity: 511.901298869
#test cost: 6.18621953616 perplexity: 486.005303387

#loading pretrained model: ./model/parameters_74837.51.pkl
#valid cost: 6.21326555195 perplexity: 499.329177169
#test cost: 6.19638286809 perplexity: 490.969922406

#loading pretrained model: ./model/parameters_89189.88.pkl
#valid cost: 6.18980438594 perplexity: 487.75068599
#test cost: 6.17049346647 perplexity: 478.422133132

#loading pretrained model: ./model/parameters_175640.91.pkl
#valid cost: 6.19320272433 perplexity: 489.411047506
#test cost: 6.17913861846 perplexity: 482.576095132

#loading pretrained model: ./model/parameters_226887.61.pkl
#valid cost: 6.1758649548 perplexity: 480.998886337
#test cost: 6.16301552196 perplexity: 474.857862283
