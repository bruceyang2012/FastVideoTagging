# FastVideoTagging 
## About this project
Fast video tagging is a project to tagging a short video (about 15-30 second time length) in less than 150 ms.
this is a application in short video understanding, for video multilabel-classification and video retrieval.
there are three main body net for fast video tagging. All the model implemented by MXNet framework.
### Basic video understanding Model 
each of above method is based on several video classification paper 
- R2Plus1D: [A Closer Look at Spatiotemporal Convolutions for Action Recognition (CVPR 2018)](https://github.com/starsdeep/R2Plus1D-MXNet)  
- MFNet:[Multi-Fiber Networks for Video Recognition](https://github.com/hudengjunai/PyTorch-MFNet)
- ECO:[ECO: Efficient Convolutional Network for Online Video Understanding](https://github.com/mzolfaghari/ECO-efficient-video-understanding)  
- C2AE:[Learning Deep Latent Spaces for Multi-Label Classification](http://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/download/14166/14487)
### Multilabel-classfication Framework
The video tagging problems is a typical Multi-label classification problems.So we choose the following MLC framework
- WARP(Weighted approximately ranking pairwise)[Deep Convolutional Ranking for Multilabel Image Annotation](https://arxiv.org/abs/1312.4894)
- LSEP(Log-sum-exp piarwise)[Improving Pairwise Ranking for Multi-label Image Classification](http://ieeexplore.ieee.org/document/8099682/)
- CNN-RNN Unified[CNN-RNN: A Unified Framework for Multi-label Image Classification,Exploring CNN-RNN Architectures for Multilabel Classiﬁcation of the Amazon](http://ieeexplore.ieee.org/document/7780620/)
- Binary Relevance(BCE) [Binary relevance for multi-label learning: an overview](http://link.springer.com/10.1007/s11704-017-7031-7)  
So we use the four kinds of loss function or framework to optimize the deep model       

## Dataset
[UCF101](http://crcv.ucf.edu/data/UCF101.php)  ：UCF101 is a typical video single label multi-classification dataset  
[Ai-Challenge 2018 FastVideoTaging]() ：The Meitu short video tagging dataset.


## DataLoader       
unlike image data loader ,the video dataloader consume a lot time if not optimized.currently state of the art video decode and load in to memory method.
- ffmpeg,just use ffmpeg to decode the key frame or frames near key frame.  
- nvvl&pynvvl,Nvidia proposed a library nvvl(nvidia video loader for abbreviation) to decode and loader video fast,there is a pytorch implementation in pynvvl,unfortunately, current nvvl does not adapt to different size and frame rate,worsely it would not free cuda memeory after fetch video sequence.  
- opencv,this is an easy way to get frames from video.just use VideoCapture to read frame.

## Result

Achieved **92.6%** Accuracy(Clip@1, prediction using only 1 clip) on UCF101 Dataset, which is **1.3% higher than the original Caffe2 model**(Accuracy 91.3%).

## Usage

### Data Preparation

#### Training 
 ```bash
$ python train.py --gpus 0,1,2,3,4,5,6,7 --pretrained ~/r2.5d_d34_l32.pkl --output ~/r2plus1d_output --batch_per_device 4 --lr 1e-4 
--model_depth 34 --wd 0.005 --num_class 101 --num_epoch 80 
```
```bash
$ python train_r3d.py --gpus 0,1 --pretrained ./r2.5d_d34_l32.pkl --output ./output --dataset meitu --loss
```   
train with loss type of Log sum exponent pairwise loss,use following command
```bash
& nohup python train_r3d.py --gpus 1 --pretrained ./output/test-0001.params --loss_type lsep_nn >mymeitu1.out 2>&1 &
```
train with loss type of weighted approximatly ranking pairwise loss,(WARP) use following command
```bash
$ nohup python train_r3d.py --gpus 1 --pretrained ./output/test-0001.params --loss_type warp_nn >mywarpnn.out 2>&1 &
```
#### Testing

Assume the training output directory is ~/r2plus1d_output and the epoch number we want to test is 80.

```
$ python validation.py --gpus 0 --output ~/r2plus1d_output --eval_epoch 80 --batch_per_device 48 --model_prefix test 
```
# The second implementation of R2+1D mxnet edition

### training and validation
```angular2html
$ python train_r3d.py --gpus 1,2 --pretrained model.params
```
## To do works
1.change the data loader to nvvl,fix the pynvvl bugs to adapted to different video size and video frame rate.  
2.add a multi-label classification loss header   
3.train a model with data meitu shot videos   
4.write the cnn-rnn unified model structure  

### origin data  
- origin train log  in /data/jh/notebooks/hudengjun/VideosFamous/R2Plus1D-MXNet

### Entry Point file
- train.py this is an implementaion for ucf101 sym writtened by Original
- train_r3d.py this is an simple-meitu and simple ucf101 dataloader train
- train_nvvl.py this is an nvvl-meitu dataloader train model
- train_unified.py this is an cnn-rnn framework train model.not implemented.

