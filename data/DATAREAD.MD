### Data Introduction
In this directory ,there are mainly two dataset defination for mxnet.UCF101 and Meitu-Ai Challenger
## UCF101
the following files are ucf101 dataset implementation
- data.py the Data.IO Iterator implementation.
- ucf101.py the mxnet.gluon.data.Dataset derived type

## Meitu-Ai Challenger
the following file are meitu dataset implentation
- simple_meitu.py the opencv dataload type
- nvvl_meitu.py the pynvvl decode nvvl dataload type
- nvvl_meitu_iter.py the pynvvl mx.io.DataIter type,batch_size =1,single thread to load data and train.for purpose of test time consume.

