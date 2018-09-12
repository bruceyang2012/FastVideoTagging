import logging
import argparse
import os
import sys
import mxnet as mx
from model import get_R2plus1d,R2Plus2D
from data import get_ucf101trainval
import numpy as np
import mxnet.gluon.loss as gloss
import mxnet.gluon.nn as nn
import mxnet.optimizer as optim
import mxnet.lr_scheduler as lr_schedualer
from mxnet import gluon
from mxnet import autograd
from time import time
import pickle
from mxnet import nd
import ipdb


def train_eval(opt):
    mx.random.seed(123)
    np.random.seed(123)
    gpus = [] if opt.gpus is None or opt.gpus is '' else [
        int(gpu) for gpu in opt.gpus.split(',')]
    num_gpus = len(gpus)
    batch_size = opt.batch_per_device*max(1,num_gpus)
    context = [mx.gpu(i) for i in gpus] if num_gpus>0 else [mx.cpu()]
    steps = [int(step) for step in opt.lr_scheduler_steps.split(',')]
    #net = get_R2plus1d(num_class=opt.num_class,model_depth=opt.model_depth)
    net = R2Plus2D(num_class=opt.num_class,model_depth=opt.model_depth)
    net.initialize(mx.init.Xavier(),
                   ctx=context)  # net parameter initialize in several cards
    if not opt.pretrained is None:
        if opt.pretrained.endswith('.pkl'):
            net.load_from_caffe2_pickle(opt.pretrained)
        elif opt.pretrained.endswith('.params'):
            net.load_from_sym_params(opt.pretrained,ctx = context)
            #load params to net context



    loss_criterion = gloss.SoftmaxCrossEntropyLoss() # loss function
    train_loader,val_loader = get_ucf101trainval(datadir=opt.datadir,
                                                 batch_size=batch_size,
                                                 n_frame=opt.n_frame,
                                                 crop_size=opt.crop_size,
                                                 scale_h=opt.scale_h,
                                                 scale_w=opt.scale_w,
                                                 num_workers=opt.num_workers) # the train and evaluation data loader
    trainer = gluon.Trainer(net.collect_params(),'sgd',
                            {'learning_rate':opt.lr,'momentum':0.9,'wd':opt.wd},
                            kvstore=opt.kvstore) # the trainer

    lr_steps = lr_schedualer.MultiFactorScheduler(steps,opt.lr_schedualer_factor)
    lr_steps.base_lr = opt.lr

    best_eval = 0.0
    for epoch in range(opt.num_epoch):
        tic = time()
        pre_loss,cumulative_loss = 0.0,0.0
        trainer.set_learning_rate(lr_steps(epoch))
        logging.info('Epoch %d learning rate %f'%(epoch,trainer.learning_rate))
        for i,(data,label) in enumerate(train_loader):
            try:
                data_list = gluon.utils.split_and_load(data,ctx_list=context,batch_axis=0)
                label_list = gluon.utils.split_and_load(label,ctx_list=context,batch_axis=0)
            except Exception as e:
                logging.info(e)
                continue
            Ls =[]
            with autograd.record():
                for x,y in zip(data_list,label_list):
                    y_hat = net(x)
                    loss = loss_criterion(y_hat,y)
                    Ls.append(loss)
                    cumulative_loss +=nd.mean(loss).asscalar()
                for L in Ls:
                    L.backward()
            trainer.step(data.shape[0])
            if (i+1)%opt.log_interval ==0:
                logging.info('[Epoch %d,Iter %d ] training loss= %f'%(
                    epoch,i+1,cumulative_loss-pre_loss
                ))
                pre_loss =cumulative_loss
                if opt.debug:
                    break
        logging.info('[Epoch %d] training loss=%f'%(epoch,cumulative_loss))
        logging.info('[Epoch %d] time used: %f'%(epoch,time()-tic))


        acc = nd.array([0],ctx=mx.cpu())
        test_iter = 0
        for i,(data,label) in enumerate(val_loader):
            try:
                data_list = gluon.utils.split_and_load(data,ctx_list=context,batch_axis=0)
                label_list = gluon.utils.split_and_load(label,ctx_list=context,batch_axis=0)
            except Exception as e:
                logging.info(e)
                continue
            for x,y in zip(data_list,label_list):
                y_hat = net(x)
                test_iter +=1
                y_pred = y_hat.argmax(axis=1)
                acc += (y_pred == y.astype('float32')).mean().asscalar() # acc in cpu
            if opt.debug:
                if i==2:
                    break
        val_acc = acc.asscalar()/test_iter
        logging.info("[Epoch %d],val acc:%f"%(epoch,val_acc))
        if val_acc>best_eval:
            net.save_parameters('./output/test-val%04d.params'%(epoch))
    logging.info("----------------finish training-----------------")



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='command for training plus 3d network')
    parser.add_argument('--gpus',type=str,default='0',help='the gpus used for training')
    parser.add_argument('--pretrained',type=str,default='',help='pretrained model parameter')
    parser.add_argument('--datadir',type=str,default='/data/jh/notebooks/hudengjun/DeepVideo/UCF-101',help='the input data directory')
    parser.add_argument('--output', type=str, default='./output/', help='the output directory')
    parser.add_argument('--optimizer',type=str,default='sgd',help='optimizer')

    parser.add_argument('--lr_scheduler_steps',type=str,default='20,40,60',help='learning rate scheduler steps')
    parser.add_argument('--lr_schedualer_factor',type=float,default=0.1,help='learning rate')
    parser.add_argument('--lr', type=float, default=1e-4, help='initialization learning rate')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay for sgd')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for sgd')
    parser.add_argument('--bn_mom', type=float, default=0.9, help='momentum for bn')
    parser.add_argument('--batch_per_device', type=int, default=4, help='the batch size')
    parser.add_argument('--batch_size', type=int, default=16, help='the batch size')
    parser.add_argument('--num_class', type=int, default=101, help='the number of class')
    parser.add_argument('--model_depth', type=int, default=34, help='network depth')
    parser.add_argument('--num_epoch', type=int, default=80, help='the number of epoch')
    parser.add_argument('--epoch_size', type=int, default=100000, help='the number of epoch')
    parser.add_argument('--begin_epoch', type=int, default=0, help='begin training from epoch begin_epoch')
    parser.add_argument('--n_frame', type=int, default=32, help='the number of frame to sample from a video')
    parser.add_argument('--crop_size', type=int, default=112, help='the size of the sampled frame')
    parser.add_argument('--scale_w', type=int, default=171, help='the rescaled width of image')
    parser.add_argument('--scale_h', type=int, default=128, help='the rescaled height of image')
    parser.add_argument('--num_workers',type=int,default=6,help='the data loader process worker number')
    parser.add_argument('--kvstore',type=str,default='device',help='kvstore to use for trainer')
    parser.add_argument('--log_interval',type=int,default=20,help='number of the batches to wait before logging')
    parser.add_argument('--debug',action='store_true',default=False)

    #parse arguments and mkdir
    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Set Logger
    logging.basicConfig(level=logging.DEBUG,
                        filename=os.path.join(args.output, 'log.txt'),
                        filemode='w')
    # Define a new Handler to log to console as well
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logging.getLogger('').addHandler(console)
    logging.info(args)
    train_eval(args)