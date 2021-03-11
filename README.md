# Introduction
This is a `pytorch_lightning` reimplementation of the Transducer module from ESPnet.
# Features
* Faster 

Experientments on timit dataset, model = RNN-T, batchsize = 4, GPU = GTX1060. A training step during an epoch cost

|Framework | Speed| 
|--|--| 
|pytorch |   6.1 step/s |
|pytorch_lightning |   6.6 step/s |
* More intuitive

Only streaming Transducer model from ESPnet is included here. Currently support
| InputLayer | Encoder | Decoder | 
|--|--| --| 
|Conv. | RNN | RNN |
| | Transformer | Transformer|
||Conformer|

# Including papers
* **[RNN-T 2012]** Graves Alex, "Sequence Transduction with Recurrent Neural Networks", 2012. 
* **[RNN-T 2013]** Graves Alex, et al., "Speech Recognition with Deep Recurrent Neural Networks", *ICASSP, IEEE International Conference on Acoustics, Speech and Signal Processing - Proceedings. 38. 10.1109/ICASSP.2013.6638947, 2013.*
* **[T-T facebook]** C.-F. Yeh, et al., "Transformertransducer: End-to-end speech recognition with self-attention", 2019.
* **[T-T google]** Q. Zhang, et al., "Transformer transducer: A streamable speech recognition model with transformer encoders and rnn-t loss", *ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), May 2020.*
* **[Conformer-T]** Gulati Anmol, et al, "Conformer: Convolution-Augmented Transformer for Speech Recognition", *Interspeech 2020, 2020, pp. 5036–5040.*

# Environment
* [kaldi](https://github.com/kaldi-asr/kaldi)  
Need to complie kaldi in advance. First clone kaldi locally, then compile `tools` and `src` according to INSTALL in their folder.
* `pip install h5py kaldiio soundfile configargparse dataclasses typeguard`

# Usage
1. Link complied kaldi to the tools directory, e.g.,` <kaldi-root>=/home/usr_name/kaldi`
```
cd ./tools
ln -s <kaldi-root>/tools
```
2. Create a new virtual environment from the system python
```
./setup_venv.sh $(command -v python3)
```
3. Configure the ceated virtual environment
```
make
```
4. Link `steps` and `utils` from wsj to the root directory
```
cd ..
ln -s <kaldi-root>/egs/wsj/s5/steps .
ln -s <kaldi-root>/egs/wsj/s5/utils .
```
5. Run
```
./run.sh 
```
6. Resume 

If the training process is accidentally interrupted, you can resume training by changing the resume variable in front of the run.sh script, e.g.,
```
resume='exp/path/to/ckpt'
```
7. Data pre-process 

The scripts of English corpus `Timit` and Mandarin corpus `Aishell-1` are already in the local directory.

# Thanks to
* [ESPnet](https://github.com/espnet/espnet)  
* [wrap rnnt](https://github.com/HawkAaron/warp-ennt)   

# Contact me 
Email: xintongwang@bjfu.edu.cn



