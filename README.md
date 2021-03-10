# Update
|Dataset | Acoustic Feature | WER(dev)| 
|--|--|--|
|timit |  fbank dim = 80  |  17.54% |

# Introduction
This is a `Transducer` Module from ESPnet and `pytorch_lightning`re-implementation 
# Features
* Faster 

Experientments on timit dataset, model = RNN-T, batchsize = 4, GPU = GTX1060 

|Framework | Speed| 
|--|--| 
|pytorch |   6.1 step/s |
|pytorch_lightning |   6.6 step/s |
* More intuitive

Only streaming Transducer model from ESPnet is included here. Currently support
| InputLayer | Encoder | Decoder | 
|--|--| --| 
|Conv. | RNN |
|pytorch_lightning |   6.6 step/s |
# Environment
* [kaldi](https://github.com/kaldi-asr/kaldi)  
Need to complie kaldi in advance. First clone kaldi locally, then compile `tools` and `src` according to INSTALL in their folder.
* requirements
```
pip install h5py kaldiio soundfile configargparse dataclasses typeguard
```
# Usage
link kaldi to tools dir, e.g., <kaldi-root>=/home/usr_name/kaldi
```
cd ./tools
ln -s <kaldi-root>/tools
```
Create a new venv from system python
```
./setup_venv.sh $(command -v python3)
```
configure venv
```
make
```
link `steps` and `utils` to root dir.
```
cd ..
ln -s <kaldi-root>/egs/wsj/s5/steps .
ln -s <kaldi-root>/egs/wsj/s5/utils .
```
## run

```
./run.sh 
```

# Thanks to
* [ESPnet](https://github.com/espnet/espnet)  
* [wrap rnnt](https://github.com/HawkAaron/warp-ennt)   

# Contect me 
Email: xintongwang@bjfu.edu.cn



