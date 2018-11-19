# coding: utf-8
import logging
import argparse
import os
import sys
import mxnet as mx
from model.R2Plus1D_hy import R2Plus2D
from data import get_ucf101trainval
from model.multi_taskR3d import R2Plus2D_MT
use_nvvl = False
from data import get_meitu_multi_task_dataloader
# from data import get_simple_meitu_dataloader
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
from model import LsepLoss, LSEP_funcLoss, WarpLoss, WARP_funcLoss, LsepLossHy


def train_eval(opt):
    mx.random.seed(123)
    np.random.seed(123)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    gpus = [] if opt.gpus is None or opt.gpus is '' else [
        int(gpu) for gpu in opt.gpus.split(',')]
    num_gpus = len(gpus)
    batch_size = opt.batch_per_device * max(1, num_gpus)
    context = [mx.gpu(i) for i in gpus][0] if num_gpus > 0 else [mx.cpu()]
    steps = [int(step) for step in opt.lr_scheduler_steps.split(',')]

    vis_env = opt.dataset + opt.output
    vis = Visulizer(env=vis_env)
    vis.log(opt)


    net = R2Plus2D_MT(num_scenes=19,num_actions=44, model_depth=opt.model_depth,
                   final_temporal_kernel=opt.n_frame // 8)  # labels set 63

    # train_loader,val_loader = get_meitu_dataloader(data_dir=opt.meitu_dir,
    #                                                device_id=opt.decoder_gpu,
    #                                                batch_size=batch_size,
    #                                                n_frame=opt.n_frame,
    #                                                crop_size=opt.crop_size,
    #                                                scale_h=opt.scale_h,
    #                                                scale_w=opt.scale_w,
    #                                                num_workers=opt.num_workers) # use multi gpus to load data
    train_loader, val_loader,sample_weight = get_meitu_multi_task_dataloader(data_dir=opt.meitu_dir,
                                                    device_id=opt.decoder_gpu,
                                                    batch_size=batch_size,
                                                    num_workers=opt.num_workers,
                                                    n_frame=opt.n_frame,
                                                    crop_size=opt.crop_size,
                                                    scale_h=opt.scale_h,
                                                    scale_w=opt.scale_w,
                                                    cache_size=opt.cache_size)

    action_loss = gloss.SoftmaxCrossEntropyLoss()

    #scene_loss = LsepLoss()

    # [type(data) for i,enumerate(train_loader) if i<2]
    # step when 66,in data/nvvl_meitu.py
    # create new find_nvv_error.py ,copy train_nvvl_r3d.py one by one test,
    # find error

    loss_dict = {'bce': gloss.SigmoidBinaryCrossEntropyLoss,
                 'warp_nn': WarpLoss,
                 'warp_fn': WARP_funcLoss,
                 'lsep_nn': LsepLoss,
                 'lsep_fn': LSEP_funcLoss}
    scene_loss = loss_dict[opt.loss_type]()

    # if opt.loss_type == 'lsep_nnh':
    #     loss_criterion = LsepLossHy(batch_size=batch_size // num_gpus, num_class=opt.num_class)
    #     loss_criterion.hybridize()
    # elif opt.loss_type == 'bce':
    #     loss_criterion = gloss.SigmoidBinaryCrossEntropyLoss()
    #     loss_criterion.hybridize()
    # else:
    #

    # net.initialize(mx.init.Xavier(),
    #                ctx=context)  # net parameter initialize in several cards

    net.initialize(mx.init.Xavier(), ctx=context)
    if not opt.pretrained is None:
        if opt.pretrained.endswith('.pkl'):
            net.load_from_caffe2_pickle(opt.pretrained)
        elif opt.pretrained.endswith('.params'):
            try:
                print("load pretrained params ", opt.pretrained)
                net.load_from_sym_params(opt.pretrained, ctx=context)
            except Exception as e:
                print("load as sym params failed,reload as gluon params")
                net.load_params(opt.pretrained, ctx=context)
                # load params to net context

    #net.hybridize()
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': opt.lr, 'momentum': 0.9, 'wd': opt.wd},
                            kvstore=opt.kvstore)  # the trainer

    lr_steps = lr_schedualer.MultiFactorScheduler(steps, opt.lr_schedualer_factor)
    lr_steps.base_lr = opt.lr

    best_eval = 0.0
    for epoch in range(opt.num_epoch):
        tic = time()
        scene_pre_loss, scene_cumulative_loss = 0.0,0.0
        action_pre_loss,action_cumulative_loss = 0.0, 0.0
        trainer.set_learning_rate(lr_steps(epoch))
        vis.log('Epoch %d learning rate %f' % (epoch, trainer.learning_rate))

        for i, (data, scene_label,action_label) in enumerate(train_loader):
            # single card not split
            with autograd.record():
                data = data.as_in_context(context)
                scene_label = scene_label.as_in_context(context)
                action_label = action_label.as_in_context(context)

                pred_scene,pred_action = net(data)
                loss_scene = scene_loss(pred_scene,scene_label)
                loss_action = action_loss(pred_action,action_label)
                loss = loss_scene + opt.action_rate*loss_action.mean()
                scene_cumulative_loss += nd.mean(loss_scene).asscalar()
                action_cumulative_loss +=nd.mean(loss_action).asscalar()
                loss.backward()
            trainer.step(data.shape[0])
            if (i + 1) % opt.log_interval == 0:
                vis.log('[Epoch %d,Iter %d ] scene loss= %f' % (epoch, i + 1, scene_cumulative_loss - scene_pre_loss))
                vis.plot('scene_loss', scene_cumulative_loss - scene_pre_loss)
                scene_pre_loss = scene_cumulative_loss

                vis.log('[Epoch %d,Iter %d ] action loss= %f' % (epoch, i + 1, action_cumulative_loss - action_pre_loss ))
                vis.plot("action_loss", action_cumulative_loss - action_pre_loss)
                action_pre_loss = action_cumulative_loss

                if opt.debug:
                    if (i + 1) // (opt.log_interval) == 3:
                        break

        vis.log('[Epoch %d] scene loss=%f,action loss=%f' % (epoch, scene_cumulative_loss,action_cumulative_loss))
        vis.log('[Epoch %d] time used: %f' % (epoch, time() - tic))
        vis.log('[Epoch %d] saving net')
        save_path = './{0}/{1}_test-val{2}.params'.format(opt.output, str(opt.dataset + 'multi'), str(epoch))
        vis.log("save path %s" % (save_path))
        net.save_parameters(save_path)



        label_inter =1e-4
        label_union =1e-4
        acc = nd.array([0], ctx=mx.cpu())
        val_iter =0
        for i,(data,scene_label,action_label) in enumerate(val_loader):
            data = data.as_in_context(context)
            action_label = action_label.as_in_context(context)
            scene_pred,action_pred = net(data)
            scene_order = scene_pred.argsort()[:,::-1]
            scene_order_np = scene_order.asnumpy()
            scene_label_np = scene_label.asnumpy()
            for scene_pred_v,scene_label_v in zip(scene_order_np,scene_label_np):
                label_set = set([index for index,value in enumerate(scene_label_v) if value>0.1])
                pred_top1 = set([scene_pred_v[0]])
                label_inter += len(pred_top1.intersection(label_set))
                label_union += len(pred_top1.union(label_set))

            action_pred = action_pred.argmax(axis=1)
            acc += (action_pred == action_label.astype('float32')).mean().asscalar()
            val_iter +=1
            if (i + 1) % (opt.log_interval) == 0:
                vis.log("[Epoch %d,Iter %d],action_acc= %f"%(epoch,i,acc.asscalar()/val_iter))
                vis.log("[Epoch %d,Iter %d],scene_top1=%f"%(epoch,i,label_inter/label_union))
                if opt.debug:
                    if (i + 1) // (opt.log_interval) == 2:
                        break
    vis.log("""----------------------------------------
               ----XXXX------finished------------------
               ----------------------------------------""")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='command for training plus 3d network')
    parser.add_argument('--gpus', type=str, default='0', help='the gpus used for training')
    parser.add_argument('--pretrained', type=str, default='./pretrain_kinetics.params',
                        help='pretrained model parameter')
    parser.add_argument('--dataset', type=str, default='meitu', help='the input data directory')
    parser.add_argument('--output', type=str, default='./output/', help='the output directory')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer')

    parser.add_argument('--lr_scheduler_steps', type=str, default='2,5,10', help='learning rate scheduler steps')
    parser.add_argument('--lr_schedualer_factor', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr', type=float, default=1e-4, help='initialization learning rate')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay for sgd')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for sgd')
    parser.add_argument('--bn_mom', type=float, default=0.9, help='momentum for bn')
    parser.add_argument('--batch_per_device', type=int, default=4, help='the batch size')
    parser.add_argument('--batch_size', type=int, default=8, help='the batch size')
    parser.add_argument('--num_class', type=int, default=63, help='the number of class')
    parser.add_argument('--model_depth', type=int, default=34, help='network depth')
    parser.add_argument('--num_epoch', type=int, default=30, help='the number of epoch')
    parser.add_argument('--epoch_size', type=int, default=10, help='the number of epoch')
    parser.add_argument('--begin_epoch', type=int, default=0, help='begin training from epoch begin_epoch')
    parser.add_argument('--n_frame', type=int, default=16, help='the number of frame to sample from a video')
    parser.add_argument('--crop_size', type=int, default=112, help='the size of the sampled frame')
    parser.add_argument('--scale_w', type=int, default=128, help='the rescaled width of image')
    parser.add_argument('--scale_h', type=int, default=128, help='the rescaled height of image')
    parser.add_argument('--num_workers', type=int, default=0, help='the data loader process worker number')
    parser.add_argument('--kvstore', type=str, default='device', help='kvstore to use for trainer')
    parser.add_argument('--log_interval', type=int, default=20, help='number of the batches to wait before logging')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--meitu_dir', type=str,
                        default='/data/jh/notebooks/hudengjun/VideosFamous/FastVideoTagging/meitu',
                        help='the meitu dataset directory')
    parser.add_argument('--loss_type', type=str, default='lsep_nn', help='the loss type for current train')
    parser.add_argument('--decoder_gpu', type=int, default=3, help='the decoder gpu to decode video to read sequence')
    parser.add_argument('--cache_size', type=int, default=20, help='the nvvl docoder lru dict cache')
    parser.add_argument('--action_rate',type=float,default=1.0,help='the loss of action rate')

    # parse arguments and mkdir
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
    """
    useage:
    python train_r3d --gpus 0,1 --pretrained ./kinectics-pretrained.params --loss_type lsep_nn --output r3d_lsep
    nohup python train_nvvl_r3d.py --gpus 3 --decoder_gpu 3 --lr 0.001 --batch_per_device 8 --n_frame 16 --loss_type bce --output nvvl_bce2 --num_workers 0 >nvvl_bce2/mytrain.out 2>&1 &
    nohup python train_multi_tasknvvl.py --gpus 3 --decoder_gpu 3 --lr 0.001 --batch_per_device 4 --n_frame 16 --loss_type lsep_nn --output multi --num_workers 0 >multi/mytrain.out 2>&1 &
    """