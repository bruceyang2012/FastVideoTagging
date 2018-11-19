from data import get_meitu_dataloader
import mxnet as mx
from model import R2Plus2D
from model import LsepLoss
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
import multiprocessing

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='the bugger findder')
    parser.add_argument('--gpus',type=str,default='0')
    parser.add_argument('--decoder_gpu',type=int,default=3)
    opt = parser.parse_args()
    #multiprocessing.set_start_method('spawn')
    print(opt)
    gpus = [] if opt.gpus is None or opt.gpus is '' else [
        int(gpu) for gpu in opt.gpus.split(',')]
    num_gpus = len(gpus)
    context = [mx.gpu(i) for i in gpus] if num_gpus > 0 else [mx.cpu()]
    train_loader,val_loader = get_meitu_dataloader(data_dir='/data/jh/notebooks/hudengjun/VideosFamous/FastVideoTagging/meitu',
                                                   device_id=opt.decoder_gpu,
                                                   batch_size=4,
                                                   num_workers=0,
                                                   n_frame=32,
                                                   crop_size=112,
                                                   scale_w=136,
                                                   scale_h=136)
    net = R2Plus2D(num_class=63,model_depth=34)
    loss_fn = LsepLoss()
    #net.initialize(mx.init.Xavier(),ctx=mx.gpu(opt.decoder_gpu))
    net.initialize(mx.init.Xavier(), ctx=context)
    for i,(data,label) in enumerate(train_loader):
        print(data.shape)

        if i==10:
            break
