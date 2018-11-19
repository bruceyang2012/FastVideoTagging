"""
paper Improving Pairwise Ranking for Multi-label Image Classification
divide the multi-label predict to two stage ：
confidence predict stage and label decision stage
confidence predict: get a rank confidence for multi-label from video sequence
decision predict:predict if the label is set from the predicted confidence,
there are two method proposed in paper ,Estimate top k or Estimate threshold
in first method.the sample output one to max(len(label))=K output .
so the multi-layer perception output K-output softmatx and the loss is CrossEntropy
in second method ,the mlp will learn a adaptive threshold for each label predict
sigmoid(f(x)-th)) is the output for each label decision.
"""
import mxnet as mx
import mxnet.gluon.nn as nn
from mxnet.gluon.loss import SigmoidBinaryCrossEntropyLoss

from model import R2Plus2D
from data import get_simple_meitu_dataloader
from model import Decision_thresh
import argparse

def train_decision(opt):
    feature_net = R2Plus2D(num_class=63,model_depth=34)
    model = Decision_thresh(thresh_size=63)
    context = mx.gpu(opt.gpu) if opt.gpu>0 else mx.cpu()

    if not opt.ranking_model is None:
        feature_net.load_params(opt.ranking_model,ctx=context)
    model.initialize(init=mx.init.Xavier(),ctx=context)
    trainer = mx.gluon.Trainer(model.collect_params(),optimizer='sgd',\
                               optimizer_params={'lr':opt.lr,'momentum':0.9,'wd':opt.wd})
    train_loader,val_loader = get_simple_meitu_dataloader()
    for epoch in range(opt.num_epoch):



if __name__=='__main__':
    #train the threshold classify
    parser = argparse.ArgumentParser(help='the train decision parser')
    parser.add_argument('--gpu',type=int,default=-1,help='the train decision model')
    parser.add_argument('--batch_size',type=int,default=4,help='the train decision model')
    parser.add_argument('--ranking_model',type=str,default='./lsep_nn/xxx.params',help='the ranking pretrained model')
    parser.add_argument('--lr',type=float,default=0.001,help='the decision model learning rate')
    parser.add_argument('--wd',type=float,default=0.00001,help='the decision model weight decay')
    parser.add_argument('--num_epoch',type=int,default=10,help='the train epoch num')

