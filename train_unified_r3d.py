# this is a unified cnn-rnn video tagging model
from model import Encoder,Decoder
import mxnet as mx
from mxnet import nd
import fire
from data import get_meitu_dataloader
import argparse
from util import Visulizer
from model import R2Plus2D
from mxnet.gluon.trainer import  Trainer
from mxnet.lr_scheduler import MultiFactorScheduler
import mxnet.gluon.loss as gloss
from time import time
from mxnet import gluon
from mxnet import autograd

def parse_basic(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    gpus = [] if opt.gpus is None or opt.gpus is '' else [
        int(gpu) for gpu in opt.gpus.split(',')]
    num_gpus = len(gpus)
    batch_size = opt.batch_per_device * max(1, num_gpus)
    context = [mx.gpu(i) for i in gpus][0] if num_gpus > 0 else [mx.cpu()]
    steps = [int(step) for step in opt.lr_scheduler_steps.split(',')]
    return batch_size,context,steps


def train_cnn(args):
    #use softmax with sigmoid cross entropy loss
    opt = args
    vis = Visulizer(env=opt.env)
    vis.log(opt)
    batch,context,steps = parse_basic(opt)
    train_loader,val_loader = get_meitu_dataloader(data_dir=opt.meitu_dir,
                                                   device_id=decoder_gpu,
                                                   batch_size=batch,
                                                   num_workers=opt.num_workers,
                                                   n_frame=opt.n_frame,
                                                   crop_size=opt.crop_size,
                                                   scale_h = opt.scale_h,
                                                   scale_w = opt.scale_w)

    net = R2Plus2D(num_class=63,model_depth=34,final_spatial_kernel=opt.crop_size//8,final_temporal_kernel=opt.n_frame//16)
    net.initialize(mx.init.Xavier(),ctx=context)
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
    lr_opts= {'learning_rate': opt.lr, 'momentum': 0.9, 'wd': opt.wd}
    trainer = Trainer(net.collect_params(),'sgd',lr_opts,kvstore=opt.kvstore)
    lr_steps = MultiFactorScheduler(steps,opt.lr_scheduler_factor)
    loss_criterion = gloss.SigmoidBinaryCrossEntropyLoss()
    for epoch in range(opt.num_epoch):
        tic = time()
        pre_loss,cumulative_loss= 0.0,0.0
        trainer.set_learning_rate(lr_steps(epoch))
        for i,(data,label) in enumerate(train_loader):
            try:
                data_list = gluon.utils.split_and_load(data,ctx_list=context,batch=0)
                label_list = gluon.utils.split_and_load(label,ctx_list=context,batch=0)
            except Exception as e:
                vis.log(e)
                continue
            Ls=[]
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
                vis.log('Epoch %d,Iter %d,Training loss=%f'%(epoch,i+1,
                                                             cumulative_loss-pre_loss))
                pre_loss = cumulative_loss
                if opt.debug:
                    break
        vis.log('[Epoch %d],trainning loss %f'%(epoch,cumulative_loss))
        vis.log('[Epoch %d],time used:%f'%(epoch,time()-tic))
        vis.log('[Epoch %d] saving net')
        save_path = './output/encoder_cnn-{0}.params'.format(str(epoch))
        vis.log('save path %d'%(save_path))
        net.save_parameters(save_path)


def train_rnn(args):
    #hold the cnn ,train the rnn
    vis = Visulizer(env=opt.env)
    vis.log(opt)
    batch, context, steps = parse_basic(opt)
    context=context[0]
    train_loader,val_loader = get_meitu_dataloader(opt.meitu_dir,
                                                   opt.decoder_gpu,
                                                   batch_size=batch,
                                                   num_workers=opt.num_workers,
                                                   n_frame=opt.n_frame,
                                                   crop_size=opt.crop_size,
                                                   scale_w=opt.scale_w,
                                                   scale_h = opt.scale_h)
    encoder = Encoder(num_class=63,model_depth=34,embed_size=200)
    state_trans = State_Trans(512,256)
    decoder = Decoder(embed_size=256,hidden_size=256,vocab_size=66,num_layers=1,max_seq_length=5)
    encoder.initialize(mx.init.Xavier(),ctx=context)
    state_trans.initialize(mx.init.Xavier(),ctx=context)
    decoder.initialize(mx.init.Xavier(),ctx=context)
    max_seq_len = opt.max_seq_len
    lr_opts = {'learning_rate': opt.lr, 'momentum': 0.9, 'wd': opt.wd}
    if not opt.encoder_pre is None:
        encoder.custom_load_params(opt.encoder_pre)
    if not opt.state_trans_pre is None:
        state_trans.load_parameters(opt.state_trans_pre)
    if not opt.decoder_pre is None:
        decoder.load_parameters(opt.decoder_pre)

    trainer1 = Trainer(state_trans.collect_params(),'sgd',lr_opts,kvstore=opt.kvstore)
    trainer2 = Trainer(decoder.collect_params(),'sgd',lr_opts,kvstore=opt.kvstore)
    lr_steps = MultiFactorScheduler(steps,factor=opt.lr_scheduler_factor)
    loss_criterion = gloss.SoftmaxCrossEntropyLoss()
    for epoch in range(opt.num_epoch):
        l_sum = 0
        tic = time()
        pre_loss, cumulative_loss = 0.0, 0.0
        trainer1.set_learning_rate(lr_steps(epoch))
        trainer2.set_learning_rate(lr_steps(epoch))
        vis.log('[Epoch %d,set learning rate'%(epoch,trainer1.learning_rate))

        for i,(data,label) in enumerate(train_loader):
            #train rnn and cnn-rnn is one context
            data = data.as_in_context(context)
            label = label.as_in_context(context)
            features = encoder(data)#type [N,C]

            with autograd.record():
                inputs = nd.ones(shape=(1,batch),ctx=context)*bos
                mask = nd.ones(shape=(1,batch),ctx=context)
                val_length = nd.array([0],ctx=context)
                feat_states = state_trans(features)
                states = decoder.begin_state(batch_size=batch,func=nd.zeros,vide_feat=feat_states)
                loss = nd.array([0],ctx=context)
                for i in range(max_seq_len):
                    y =label[i]
                    outputs,states = decoder(inputs,states)
                    #outputs shape is 1NC,states is list of [LNC]
                    inputs = outputs.argmax(axis=2) # shape is 1xN just for annother input
                    val_length = val_length +mask.sum()
                    outputs = outputs.reshape(batch,-1)
                    loss = loss + (loss_criterion(outputs,y)*mask).sum()
                    mask = mask * (inputs != eos)
                loss = loss/val_length
                loss.backward()
                trainer1.step(1)
                trainer2.step(1)
            l_sum += loss.asscalar()

            if (i+1)%(opt.log_interval)==0:
                vis.log('Epoch %d,Iter %d,Training loss=%f' % (epoch, i + 1,
                                                               cumulative_loss - pre_loss))
                pre_loss = cumulative_loss
                if opt.debug:
                    break
        # for one epoch







def train_rnn_cnn(args):
    #train the cnn and rnn simu
    vis = Visulizer(env=opt.env)
    vis.log(opt)
    batch, context, steps = parse_basic(opt)
    context = context[0]
    train_loader, val_loader = get_meitu_dataloader(opt.meitu_dir,
                                                    opt.decoder_gpu,
                                                    batch_size=batch,
                                                    num_workers=opt.num_workers,
                                                    n_frame=opt.n_frame,
                                                    crop_size=opt.crop_size,
                                                    scale_w=opt.scale_w,
                                                    scale_h=opt.scale_h)

    encoder =




if __name__=='__main__':
    parser = argparse.ArgumentParser(description='command for training plus 3d network')
    parser.add_argument('--gpus', type=str, default='0', help='the gpus used for training')
    parser.add_argument('--pretrained', type=str, default='./pretrain_kinetics.params',
                        help='pretrained model parameter')
    parser.add_argument('--dataset', type=str, default='meitu', help='the input data directory')
    parser.add_argument('--env', type=str, default='crnn', help='the output directory')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer')

    parser.add_argument('--lr_scheduler_steps', type=str, default='2,5,10', help='learning rate scheduler steps')
    parser.add_argument('--lr_scheduler_factor', type=float, default=0.1, help='learning rate')
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
    parser.add_argument('--loss_type', type=str, default='lsep_nnh', help='the loss type for current train')
    parser.add_argument('--decoder_gpu', type=int, default=3, help='the decoder gpu to decode video to read sequence')
    parser.add_argument('--cache_size', type=int, default=20, help='the nvvl docoder lru dict cache')
    parser.add_argument('--encoder_pre',type=str,default='./checkpoint/test-0001.params',help='the encoder cnn params')
    parser.add_argument('--state_trans_pre',type=str,default='./checkpoint/state-0001.paras',help='the hidden state trans params')
    parser.add_argument('--decoder_pre',type=str,default='./checkpoint/decoder-0001.param',help='the decoder parameters')
    parser.add_argument('--max_seq_len',type=int,default=5,help='the max seq len of labels list')
    # parse arguments and mkdir
    args = parser.parse_args()

    # Set Logger
    logging.basicConfig(level=logging.DEBUG,
                        filename=os.path.join('./output',args.env+'.txt'),
                        filemode='w')
    # Define a new Handler to log to console as well
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logging.getLogger('').addHandler(console)
    logging.info(args)

    fire.Fire()


    # encoder = Encoder(num_class=62,model_depth=34,embed_size=100)
    # decoder = Decoder(embed_size=200,hidden_size=256,vocab_size=64,num_layers=2,max_seq_length=5)
    #
    # encoder.initialize(mxnet.init.Xavier())
    # decoder.initialize(mxnet.init.Xavier())
    #
    #
    # video_clip = nd.random.normal(shape=(4,3,32,112,112))
    # video_feature  = encoder(video_clip)
    # states = decoder.begin_state(batch_size=4,func=nd.zeros,vid_feat=video_feature)
    #
    # inputs = nd.array([[1,1,1,1]])
    # outputs,states = decoder(inputs,states)
    # print(outputs.shape,states[0].shape)