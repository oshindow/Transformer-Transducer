# Update
|Dataset | Acoustic Feature | Word Embedding | PER(dev)| 
|--|--|--|--|
|thchs30 |  fbank dim = 40  |    one-hot     |  17.54% |

# Introduction
This directory contains a `pytorch` implementation of 
>Transformer Transducer: A streamable speech recognition model with transformer encoders and RNN-T loss

which is pre-printed on arXiv in Feb. 2020 from Google. It shows that Transformer Transducer model achieved state-of-the-art results in streaming speech recognition.

# Features
Transformer Transducer (T-T) is a combination of Transformer and RNN-T, which employs self-attention [1] to encode both acoustic features and word embeddings respectively instead of LSTM in RNN Transducer. Not only T-T uses the Relative Positional Encoding, which is mentioned in transformer-xl [2], but also Loss Function [3] and Joint Network [4] proposed by Alex Graves in 2012 and 2013 respectively. 

# Environment
* [Kaldi](https://github.com/kaldi-asr/kaldi)  
Use Kaldi as a toolbox to extract the MFCC (dim=39) or Fbank (dim=40) features
* pytorch >=0.4
* [wraprnnt](https://github.com/HawkAaron/warp-transducer)  
which is the wrapped RNNT Loss function

# Usage
## train
Before start to train, make sure that you already get the acoustic features for {train, dev} and the model unit either character or words, which may not yet supported in the origin dataset.
```
python train.py 
```

# Thanks to
* [rnn-t](https://github.com/ZhengkunTian/rnn-transducer)  
* [wrap rnnt](https://github.com/HawkAaron/warp-transducer)  
* [transformer-xl](https://github.com/kimiyoung/transformer-xl)  

# Reference
[1] [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf)  
[2] [Transformer-xl: Attentive language models beyond a fixed-length context](https://arxiv.org/pdf/1901.02860.pdf)  
[3] [Sequence transduction with recurrent neural networks](http://www.cs.toronto.edu/~graves/icml_2012.pdf)  
[4] [Speech recognition with deep recurrent neural networks](http://www.cs.toronto.edu/~fritz/absps/RNN13.pdf)  

# Author
Email: walston874848612@163.com



