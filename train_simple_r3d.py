import logging
import argparse
import os
import sys
import mxnet as mx
from model import R2Plus2D
from data import get_ucf101trainval
use_nvvl=False
#from data import get_meitu_dataloader
from data import get_simple_meitu_dataloader
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
import time as tmm
import ipdb
from util import Visulizer
from model import LsepLoss,LSEP_funcLoss,WarpLoss,WARP_funcLoss
from model import Decision_thresh
def train_eval(opt):
    mx.random.seed(123)
    np.random.seed(123)
    os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
    gpus = [] if opt.gpus is None or opt.gpus is '' else [
        int(gpu) for gpu in opt.gpus.split(',')]
    num_gpus = len(gpus)
    batch_size = opt.batch_per_device*max(1,num_gpus)
    context = [mx.gpu(i) for i in gpus] if num_gpus>0 else [mx.cpu()]
    steps = [int(step) for step in opt.lr_scheduler_steps.split(',')]
    vis_env = opt.dataset + opt.output
    vis = Visulizer(env=vis_env)
    vis.log(opt)

    #optional ucf101 or meitu,get net structure,loss criterion,train val loader
    if opt.dataset=='ucf101' or opt.dataset=='ucf':
        net = R2Plus2D(num_class=101,model_depth=opt.model_depth)
        loss_criterion = gloss.SoftmaxCrossEntropyLoss() # loss function
        train_loader, val_loader = get_ucf101trainval(datadir='/data/jh/notebooks/hudengjun/DeepVideo/UCF-101',
                                                      batch_size=batch_size,
                                                      n_frame=opt.n_frame,
                                                      crop_size=opt.crop_size,
                                                      scale_h=opt.scale_h,
                                                      scale_w=opt.scale_w,
                                                      num_workers=opt.num_workers)  # the train and evaluation data loader
    elif opt.dataset =='meitu':
        net = R2Plus2D(num_class=opt.num_class,model_depth=opt.model_depth) # labels set 63

        # loss_criterion = LsepLoss()
        # loss_criterion = gloss.SigmoidBinaryCrossEntropyLoss()
        # train_loader,val_loader = get_meitu_dataloader(data_dir=opt.meitu_dir,
        #                                                device_ids=gpus,
        #                                                n_frame=opt.n_frame,
        #                                                crop_size=opt.crop_size,
        #                                                scale_h=opt.scale_h,
        #                                                scale_w=opt.scale_w,
        #                                                num_workers=opt.num_workers) # use multi gpus to load data
        train_loader,val_loader = get_simple_meitu_dataloader(datadir=opt.meitu_dir,
                                                              batch_size=batch_size,
                                                              n_frame=opt.n_frame,
                                                              crop_size=opt.crop_size,
                                                              scale_h=opt.scale_h,
                                                              scale_w=opt.scale_w,
                                                              num_workers=opt.num_workers)
    loss_dict = {'bce':gloss.SigmoidBinaryCrossEntropyLoss,
                 'warp_nn':WarpLoss,
                 'warp_fn':WARP_funcLoss,
                 'lsep_nn':LsepLoss,
                 'lsep_fn':LSEP_funcLoss}
    if opt.loss_type=='bce':
        loss_criterion = gloss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
    else:
        loss_criterion = loss_dict[opt.loss_type]()


    net.initialize(mx.init.Xavier(),
                   ctx=context)  # net parameter initialize in several cards
    if not opt.pretrained is None:
        if opt.pretrained.endswith('.pkl'):
            net.load_from_caffe2_pickle(opt.pretrained)
        elif opt.pretrained.endswith('.params'):
            try:
                print("load pretrained params ",opt.pretrained)
                net.load_from_sym_params(opt.pretrained,ctx = context)
            except Exception as e:
                print("load as sym params failed,reload as gluon params")
                net.load_params(opt.pretrained,ctx=context)
                #load params to net context

    trainer = gluon.Trainer(net.collect_params(),'sgd',
                            {'learning_rate':opt.lr,'momentum':0.9,'wd':opt.wd},
                            kvstore=opt.kvstore) # the trainer

    lr_steps = lr_schedualer.MultiFactorScheduler(steps,opt.lr_scheduler_factor)
    lr_steps.base_lr = opt.lr

    best_eval = 0.0
    for epoch in range(opt.num_epoch):
        tic = time()
        pre_loss,cumulative_loss = 0.0,0.0
        trainer.set_learning_rate(lr_steps(epoch))
        vis.log('Epoch %d learning rate %f'%(epoch,trainer.learning_rate))
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
                vis.log('[Epoch %d,Iter %d ] training loss= %f'%(
                    epoch,i+1,cumulative_loss-pre_loss
                ))
                pre_loss =cumulative_loss
                if opt.debug:
                    break
        vis.log('[Epoch %d] training loss=%f'%(epoch,cumulative_loss))
        vis.log('[Epoch %d] time used: %f'%(epoch,time()-tic))
        vis.log('[Epoch %d] saving net')
        save_path = './{0}/{1}_test-val{2}.params'.format(opt.output,str(opt.dataset + opt.loss_type), str(epoch))
        vis.log("save path %s"%(save_path))
        net.save_parameters(save_path)

        best_iou=0.0
        if opt.dataset=='ucf101' or opt.dataset =='ucf':
            acc = nd.array([0],ctx=mx.cpu())
            test_iter = 0
            for i,(data,label) in enumerate(val_loader):
                try:
                    data_list = gluon.utils.split_and_load(data,ctx_list=context,batch_axis=0)
                    label_list = gluon.utils.split_and_load(label,ctx_list=context,batch_axis=0)
                except Exception as e:
                    logging.info(e)
                    continue
                #try to get data and label,then try to flow data in net

                try:
                    for x,y in zip(data_list,label_list):
                        y_hat = net(x)
                        test_iter +=1 # single iter
                        y_pred = y_hat.argmax(axis=1)
                        acc += (y_pred == y.astype('float32')).mean().asscalar() # acc in cpu
                except Exception as e:
                    logging.info(e)
                    continue

                if opt.debug:
                    if i==2:
                        break
                val_acc = acc.asscalar() / test_iter
                if (i+1) %(opt.log_interval)==0:
                    logging.info("[Epoch %d,Iter %d],acc=%f" % (epoch,i,val_acc))
        elif opt.dataset=='meitu':
            k=4
            topk_inter = np.array([1e-4]*k)
            topk_union = np.array([1e-4]*k)

            for i,(data,label) in enumerate(val_loader):
                try:
                    data_list = gluon.utils.split_and_load(data,ctx_list=context,batch_axis=0)
                    label_list = gluon.utils.split_and_load(label,ctx_list=context,batch_axis=0)
                except Exception as e:
                    logging.info(e)
                    continue
                try:
                    for x,y in zip(data_list,label_list):
                        y_hat = net(x)
                        pred_order = y_hat.argsort()[:,::-1] # sort and desend order
                        #just compute top1 label
                        pred_order_np = pred_order.asnumpy()
                        y_np = y.asnumpy()
                        if opt.debug:
                            print("pred shape and target shape",pred_order_np.shape,y_np.shape)
                        for pred_vec,y_vec in zip(pred_order_np,y_np):
                            label_set =set([index for index,value in enumerate(y_vec) if value>0.1])
                            pred_topk = [set(pred_vec[0:k]) for k in range(1,k+1)]
                            topk_inter +=np.array([len(p_k.intersection(label_set)) for p_k in pred_topk])
                            topk_union +=np.array([len(p_k.union(label_set)) for p_k in pred_topk])
                    if (i+1) %(opt.log_interval)==0:
                        vis.log("[Epoch %d,Iter %d],time %s,Iou %s" % (epoch, i, \
                                                                            tmm.strftime("%Y-%D:%H-%S"), \
                                                                            str(topk_inter / topk_union)))
                except Exception as e:
                    logging.info(e)

                if opt.debug and not opt.rand_test:
                    if i==2:
                        break
                #closure for try an iter of test.

    vis.log("""----------------------------------------
               ----XXXX------finished------------------
               ----------------------------------------""")

def train_decision(opt):
    mx.random.seed(123)
    np.random.seed(123)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    gpus = [] if opt.gpus is None or opt.gpus is '' else [
        int(gpu) for gpu in opt.gpus.split(',')]
    num_gpus = len(gpus)
    batch_size = opt.batch_per_device * max(1, num_gpus)
    context = [mx.gpu(i) for i in gpus] if num_gpus > 0 else [mx.cpu()]
    steps = [int(step) for step in opt.lr_scheduler_steps.split(',')]

    feature_net = R2Plus2D(num_class=62,model_depth=34)
    model = Decision_thresh(thresh_size=62)

    if not opt.ranking_model is None:
        feature_net.load_params(opt.ranking_model,ctx=context)
    model.initialize(init=mx.init.Xavier(),ctx=context)
    trainer = mx.gluon.Trainer(model.collect_params(),'sgd',\
                               {'learning_rate':opt.lr,'momentum':0.9,'wd':opt.wd},
                               kvstore=opt.kvstore)
    train_loader, val_loader = get_simple_meitu_dataloader(datadir=opt.meitu_dir,
                                                           batch_size=batch_size,
                                                           n_frame=opt.n_frame,
                                                           crop_size=opt.crop_size,
                                                           scale_h=opt.scale_h,
                                                           scale_w=opt.scale_w,
                                                           num_workers=opt.num_workers)
    loss_criterion = gluon.loss.SigmoidBinaryCrossEntropyLoss()
    lr_steps = lr_schedualer.MultiFactorScheduler(steps,opt.lr_scheduler_factor)
    best_eval = 0.0
    for epoch in range(opt.num_epoch):
        tic = time()
        pre_loss,cumulative_loss = 0.0,0.0
        trainer.set_learning_rate(lr_steps(epoch))
        logging.info('Epoch %d learning rate %f to make decision through threshold'%(epoch,trainer.learning_rate))
        for i,(data,label) in enumerate(train_loader):
            try:
                data_list = gluon.utils.split_and_load(data,ctx_list=context,batch_axis=0)
                label_list = gluon.utils.split_and_load(label,ctx_list=context,batch_axis=0)
            except Exception as e:
                logging.info(e)
                continue
            Ls =[]
            confidences = []
            for x in data_list:
                confidences.append(feature_net(x))
            with autograd.record():
                Ls=[]
                for conf,y in zip(confidences,label_list):
                    decision = model(conf)
                    loss = loss_criterion(decision,y)
                    Ls.append(loss)
                    cumulative_loss += nd.mean(loss).asscalar()
                for L in Ls:
                    L.backward()
            trainer.step(data.shape[0])
            if (i+1)%opt.log_interval ==0:
                logging.info('[Epoch %d,Iter %d] ,training loss=%f'%(
                    epoch+1,i+1,cumulative_loss-pre_loss))
                pre_loss = cumulative_loss
                print(model.collect_params()['decision_thresh0_thresh'].data())
            if opt.debug:
                break
        logging.info('[Epoch %d] training loss = %f'%(epoch,cumulative_loss))
        logging.info('[Epoch %d] time used:%f'%(epoch,time()-tic))
        logging.info('[Epoch %d] save net')
        model.save_parameters('./{0}/{1}_decisionmodel_{2}.params'.format(opt.output,str(opt.dataset+opt.loss_type),str(epoch)))

        # begin to evaluation the model

        inter = 1e-4
        union = 1e-4
        tic = time()
        for i,(data,label) in enumerate(val_loader):
            try:
                data_list = gluon.utils.split_and_load(data,ctx_list=context,batch_axis=0)
                label_list = gluon.utils.split_and_load(label,ctx_list=context,batch_axis=0)
            except Exception as e:
                logging.info(e)
                continue
            try:
                for x,y in zip(data_list,label_list):
                    conf = feature_net(x)
                    sig_label = model(conf)
                    y_np = y.asnumpy()
                    sig_np = sig_label.asnumpy()
                    rows,indexs = np.where(sig_np>0.5)
                    labelset_list = []
                    for j in range(x.shape[0]):
                        labelset_list.append(set())
                    #labelset_list = [set()]*x.shape[0] # the sample number
                    for row ,index in zip(rows,indexs):
                        labelset_list[row].add(index)
                    for pred_set,gt_vec in zip(labelset_list,y_np):
                        gt_set = set([index for index,value in enumerate(gt_vec) if value>0.1])
                        inter += len(pred_set.intersection(gt_set))
                        union += len(pred_set.union(gt_set))
            except Exception as e:
                print(e)
                continue
            if (i+1) %(opt.log_interval)==0:
                logging.info('[Epoch %d,Iter %d],time %s,IoU %s'%(epoch,i,tmm.strftime("%Y-%D:%H-%S"),str(inter/union)))
                if opt.debug:
                    break
        logging.info("finish one epoch validataion")
        logging.info("[Epoch %d],validation time used %d"%(epoch,time()-tic))
    logging.info("finish all epoch trainning and test")



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='command for training plus 3d network')
    parser.add_argument('--gpus',type=str,default='0',help='the gpus used for training')
    parser.add_argument('--pretrained',type=str,default='',help='pretrained model parameter')
    parser.add_argument('--dataset',type=str,default='meitu',help='the input data directory')
    parser.add_argument('--output', type=str, default='./output/', help='the output directory')
    parser.add_argument('--optimizer',type=str,default='sgd',help='optimizer')

    parser.add_argument('--lr_scheduler_steps',type=str,default='2,5,10',help='learning rate scheduler steps')
    parser.add_argument('--lr_scheduler_factor',type=float,default=0.1,help='learning rate')
    parser.add_argument('--lr', type=float, default=1e-4, help='initialization learning rate')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay for sgd')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for sgd')
    parser.add_argument('--bn_mom', type=float, default=0.9, help='momentum for bn')
    parser.add_argument('--batch_per_device', type=int, default=4, help='the batch size')
    parser.add_argument('--batch_size', type=int, default=4, help='the batch size')
    parser.add_argument('--num_class', type=int, default=63, help='the number of class')
    parser.add_argument('--model_depth', type=int, default=34, help='network depth')
    parser.add_argument('--num_epoch', type=int, default=30, help='the number of epoch')
    parser.add_argument('--epoch_size', type=int, default=100000, help='the number of epoch')
    parser.add_argument('--begin_epoch', type=int, default=0, help='begin training from epoch begin_epoch')
    parser.add_argument('--n_frame', type=int, default=16, help='the number of frame to sample from a video')
    parser.add_argument('--crop_size', type=int, default=112, help='the size of the sampled frame')
    parser.add_argument('--scale_w', type=int, default=128, help='the rescaled width of image')
    parser.add_argument('--scale_h', type=int, default=128, help='the rescaled height of image')
    parser.add_argument('--num_workers',type=int,default=6,help='the data loader process worker number')
    parser.add_argument('--kvstore',type=str,default='device',help='kvstore to use for trainer')
    parser.add_argument('--log_interval',type=int,default=20,help='number of the batches to wait before logging')
    parser.add_argument('--debug',action='store_true',default=False)
    parser.add_argument('--rand_test',action='store_true',default=False)
    parser.add_argument('--meitu_dir',type=str,default='/data/jh/notebooks/hudengjun/meitu',help='the meitu dataset directory')
    parser.add_argument('--loss_type',type=str,default='lsep_nn',help='the loss type for current train')
    parser.add_argument('--sample_stride',type=int,default=1,help='the opencv decoder stride')
    parser.add_argument('--train_decision',action='store_true',default=False)
    parser.add_argument('--ranking_model',type=str,default='./lsep_nn2/meitulsep_nn_test-val0.params',help='the trained confidence ranking model')
    #parser.add_argument('--')
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
    if not args.train_decision:
        train_eval(args)
    else:
        train_decision(args)
    """
    useage 1 train r3d ranking model:
    python train_r3d --gpus 0,1 --pretrained ./output/test-0017.params --loss_type warp_nn --output ./hpc4
    
    useage 2 train label_decision model:
    ptyon train_simple_r3d --gpus 0,1 --ranking_model ./lsep_nn/test_.params --lr 0.001 --num_epoch 5 --output ./lsep_nn --train_decision
    nohup python train_simple_r3d.py --gpus 1 --ranking_model ./lsep_nn2/meitulsep_nn_test-val1.params --output ./decision2 --train_decision > ./decision2/mydecision.out 2>&1
    """