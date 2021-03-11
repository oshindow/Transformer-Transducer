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
If the training process is accidentally interrupted, you can resume training through resume by changing the resume variable in front of the run.sh script, e.g.,
```
resume='exp/path/to/ckpt'
```
7. Data pre-process
The scripts of English corpus `Timit`and Mandrian corpus `Aishell-1` are already in the local directory.

# Thanks to
* [ESPnet](https://github.com/espnet/espnet)  
* [wrap rnnt](https://github.com/HawkAaron/warp-ennt)   

# Contact me 
Email: xintongwang@bjfu.edu.cn



