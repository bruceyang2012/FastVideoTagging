from mxnet import nd
from mxnet import sym
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import optimizer as optim
from mxnet.gluon import loss as gloss
import os
import sys
import mxnet as mx
from time import time
class ModelBuilder():
    def __init__(self,no_bias,bn_mom=0.9):
        self.comp_count=0
        self.comp_idx=0
        self.bn_mom = bn_mom
        self.no_bias = 1 if no_bias else 0


def get_spatial_temporal_conv(in_filters,out_filter,stride,use_bias=False):
    blk = nn.Sequential()

    i = 3*in_filters*out_filter*3*3
    i /= in_filters*3*3 + 3*out_filter
    middle_filters = int(i)
    #print("Number of middle filters: {0}".format(middle_filters))
    blk.add(
        nn.Conv3D(channels=middle_filters,
                  kernel_size=(1,3,3),
                  strides=(1,stride[0],stride[1]),
                  padding=(0,1,1),
                  use_bias=use_bias),
        nn.BatchNorm(),
        nn.Activation(activation='relu'),
        nn.Conv3D(channels=out_filter,
                  kernel_size=(3,1,1),
                  strides=(stride[0],1,1),
                  padding=(1,0,0),
                  use_bias=use_bias)
    )
    return blk

class R3DBlock(nn.Block):
    def __init__(self,
                 input_filter,
                 num_filter,
                 comp_index=-1,
                 downsampling=False,
                 spation_batch_norm=True,
                 only_spatial_downsampling=False,
                 use_bias = False):
        super(R3DBlock,self).__init__()
        if comp_index is -1:
            print("error construct a residual block")
        if downsampling:
            self.use_striding = [1,2,2] if only_spatial_downsampling else [2,2,2]
        else:
            self.use_striding= [1,1,1]
        self.spatial_temporal_conv1 = get_spatial_temporal_conv(input_filter,num_filter,self.use_striding,use_bias=use_bias)
        self.bn1 = nn.BatchNorm()
        self.relu1 = nn.Activation(activation='relu')
        self.spatial_temporal_conv2 = get_spatial_temporal_conv(num_filter,num_filter,stride=[1,1,1],use_bias=use_bias)
        self.bn2 = nn.BatchNorm()
        self.num_filter = num_filter
        self.input_filter = input_filter
        self.downsampling = downsampling
        if num_filter != input_filter or downsampling:
            self.branch_conv = nn.Conv3D(channels=num_filter,
                                         kernel_size=[1,1,1],
                                         strides=self.use_striding,
                                         use_bias=use_bias)
            self.branch_bn = nn.BatchNorm()

    def forward(self, x):
        y = self.spatial_temporal_conv1(x)
        y = self.relu1(self.bn1(y))
        y = self.spatial_temporal_conv2(y)
        y = self.bn2(y)
        if self.num_filter != self.input_filter or self.downsampling:
            x = self.branch_conv(x)
            x = self.branch_bn(x)
        out =nd.relu(y+x)
        return out

BLOCK_CONFIG = {
    10: (1, 1, 1, 1),
    16: (2, 2, 2, 1),
    18: (2, 2, 2, 2),
    26: (2, 3, 4, 3),
    34: (3, 4, 6, 3),
}


class R2Plus2D(nn.Block):
    def __init__(self,num_class,model_depth,final_spatial_kernel=7,final_temporal_kernel=4,with_bias=False):
        super(R2Plus2D,self).__init__()
        self.comp_count=0
        self.base = nn.Sequential(prefix='base_')
        with self.base.name_scope():
            self.base.add(
                nn.Conv3D(channels=45,
                      kernel_size=(1, 7, 7),
                      strides=(1, 2, 2),
                      padding=(0, 3, 3),
                      use_bias=with_bias),
                nn.BatchNorm(),
                nn.Activation(activation='relu'),
                nn.Conv3D(channels=64,
                      kernel_size=(3, 1, 1),
                      strides=(1, 1, 1),
                      padding=(1, 0, 0),
                      use_bias=with_bias),
                nn.BatchNorm(),
                nn.Activation(activation='relu')
            )

        self.base_name = self.set_base_name()
        (n2, n3, n4, n5) = BLOCK_CONFIG[model_depth]


        self.conv2_name =[]
        self.conv2 = nn.Sequential(prefix='conv2_')
        with self.conv2.name_scope():
            for _ in range(n2):
                self.conv2_name.extend(self.add_comp_count_index(change_channels=False,comp_index=self.comp_count,prefix=self.conv2.prefix))
                self.conv2.add(R3DBlock(input_filter=64, num_filter=64, comp_index=self.comp_count,use_bias=with_bias))
                self.comp_count += 1

        #self.conv3
        self.conv3_name = []
        self.conv3 = nn.Sequential(prefix='conv3_')
        with self.conv3.name_scope():
            print("this in conv3 comp_count is ",self.comp_count)
            self.conv3_name.extend(self.add_comp_count_index(change_channels=True,downsampling=True,comp_index=self.comp_count))
            self.conv3.add(R3DBlock(input_filter=64, num_filter=128, comp_index=self.comp_count, downsampling=True,use_bias=with_bias))

            self.comp_count += 1
            for _ in range(n3 - 1):
                self.conv3_name.extend(self.add_comp_count_index(change_channels=False,downsampling=False,comp_index=self.comp_count))
                self.conv3.add(R3DBlock(input_filter=128, num_filter=128, comp_index=self.comp_count,use_bias=with_bias))
                self.comp_count += 1

        # self.conv4
        self.conv4_name=[]
        self.conv4 = nn.Sequential(prefix='conv4_')
        with self.conv4.name_scope():
            self.conv4_name.extend(self.add_comp_count_index(change_channels=True,downsampling=True, comp_index=self.comp_count))
            self.conv4.add(R3DBlock(128, 256, comp_index=self.comp_count, downsampling=True,use_bias=with_bias))
            self.comp_count += 1

            for _ in range(n4 - 1):
                self.conv4_name.extend(self.add_comp_count_index(change_channels=False,downsampling=False,comp_index=self.comp_count))
                self.conv4.add(R3DBlock(256, 256,comp_index=self.comp_count,use_bias=with_bias))
                self.comp_count += 1

        #conv5
        self.conv5_name  = []
        self.conv5 = nn.Sequential(prefix='conv5_')
        with self.conv5.name_scope():
            self.conv5_name.extend(self.add_comp_count_index(change_channels=True,downsampling=True, comp_index=self.comp_count))
            self.conv5.add(R3DBlock(256, 512, comp_index=self.comp_count, downsampling=True,use_bias=with_bias))
            self.comp_count +=1
            for _ in range(n5 - 1):
                self.conv5_name.extend(self.add_comp_count_index(comp_index=self.comp_count))
                self.conv5.add(R3DBlock(512, 512, self.comp_count,use_bias=with_bias))
                self.comp_count += 1

        # final output of conv5 is [512,t/8,7,7]
        self.avg = nn.AvgPool3D(pool_size=(final_temporal_kernel, final_spatial_kernel, final_spatial_kernel),
                             strides=(1, 1, 1),
                             padding=(0, 0, 0))
        self.output = nn.Dense(units=num_class)
        self.dense0_name = ['final_fc_weight','final_fc_bias']

    @staticmethod
    def set_base_name():
        base_name = ['conv1_middle_weight',
                          'conv1_middle_spatbn_relu_gamma',
                          'conv1_middle_spatbn_relu_beta',
                          'conv1_middle_spatbn_relu_moving_mean',
                          'conv1_middle_spatbn_relu_moving_var',
                          'conv1_weight',
                          'conv1_spatbn_relu_gamma',
                          'conv1_spatbn_relu_beta',
                          'conv1_spatbn_relu_moving_mean',
                          'conv1_spatbn_relu_moving_var']
        return base_name

    @staticmethod
    def add_comp_count_index(change_channels=False,downsampling=False,comp_index=-1,prefix=None):
        #add spatial_temporal_conv
        conv_list =[]

        conv_idx = 1
        conv_list.extend(['comp_%d_conv_%d_middle_weight'%(comp_index,conv_idx),
                          'comp_%d_spatbn_%d_middle_gamma'%(comp_index,conv_idx),
                          'comp_%d_spatbn_%d_middle_beta' % (comp_index, conv_idx),
                          'comp_%d_spatbn_%d_middle_moving_mean' % (comp_index, conv_idx),
                          'comp_%d_spatbn_%d_middle_moving_var' % (comp_index, conv_idx),
                          'comp_%d_conv_%d_weight'%(comp_index,conv_idx)])

        conv_list.extend(['comp_%d_spatbn_%d_gamma'%(comp_index,conv_idx),
                          'comp_%d_spatbn_%d_beta' % (comp_index, conv_idx),
                          'comp_%d_spatbn_%d_moving_mean' % (comp_index, conv_idx),
                          'comp_%d_spatbn_%d_moving_var' % (comp_index, conv_idx)])

        #add spatial_temporal_conv
        conv_idx +=1
        conv_list.extend(['comp_%d_conv_%d_middle_weight' % (comp_index, conv_idx),
                          'comp_%d_spatbn_%d_middle_gamma' % (comp_index, conv_idx),
                          'comp_%d_spatbn_%d_middle_beta' % (comp_index, conv_idx),
                          'comp_%d_spatbn_%d_middle_moving_mean' % (comp_index, conv_idx),
                          'comp_%d_spatbn_%d_middle_moving_var' % (comp_index, conv_idx),
                          'comp_%d_conv_%d_weight' % (comp_index, conv_idx)])

        conv_list.extend(['comp_%d_spatbn_%d_gamma' % (comp_index, conv_idx),
                          'comp_%d_spatbn_%d_beta' % (comp_index, conv_idx),
                          'comp_%d_spatbn_%d_moving_mean' % (comp_index, conv_idx),
                          'comp_%d_spatbn_%d_moving_var' % (comp_index, conv_idx)])

        # check to add the extra layer
        if change_channels or downsampling:
            conv_list.extend(['shortcut_projection_%d_weight'%(comp_index),
                              'shortcut_projection_%d_spatbn_gamma'%(comp_index),
                              'shortcut_projection_%d_spatbn_beta' % (comp_index),
                              'shortcut_projection_%d_spatbn_moving_mean' % (comp_index),
                              'shortcut_projection_%d_spatbn_moving_var' % (comp_index)])
        return conv_list




    def forward(self, x):
        x = self.base(x)
        #print('after base',x.shape)
        x = self.conv2(x)
        #print('after conv2',x.shape)
        x = self.conv3(x)
        #print('after conv3',x.shape)
        x = self.conv4(x)
        #print('after conv4',x.shape)
        x = self.conv5(x)
        #print('after conv5',x.shape)
        x = self.avg(x)
        #print('after avg',x.shape)
        return self.output(x)

    def extract_features(self,x):
        x = self.base(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avg(x)
        return x

    def load_from_sym_params(self,f,ctx = mx.cpu()):
        """f is a file name store by mxnet params"""
        if not os.path.exists(f):
            print("parameter file is not exist",f)
            return
        sym_dict = nd.load(f)

        trans_dict ={}
        print(type(sym_dict))
        for k,v in sym_dict.items():
            trans_dict[k.split(':')[-1]] = v

        for cld in self._children.values():
            if cld.name=='pool0':
                continue
            params = cld.collect_params()
            model_layer_keys = params.keys()
            layer_name = cld.name+'_name'
            stored_keys = self.__getattribute__(layer_name)
            for model_k,connect_k in zip(model_layer_keys,stored_keys):
                #print(model_k,connect_k)
                if model_k.startswith('dense0'):
                    continue
                params[model_k]._load_init(trans_dict[connect_k],ctx=ctx)





    def load_from_caffe2_pickle(self,f):
        """f is a caffe2 pickle file trained from R2PLUS1D"""
        if os.path.exists(f):
            print("parameter file is not exist",f)
        params = self._collect_params_with_prefix()
        caffe2_dicts = pickle.load(open(f, 'rb'), encoding='latin1')['blobs']




def get_R2plus1d(num_class=101,
                 no_bias=0,
                 model_depth=18,
                 final_spatial_kernel=7,
                 final_temporal_kernel=4):
    comp_count=0
    net = nn.Sequential()
    net.add(
        nn.Conv3D(channels=45,
                      kernel_size=(1,7,7),
                      strides=(1,2,2),
                      padding=(0,3,3)),
        nn.BatchNorm(),
        nn.Activation(activation='relu'),
        nn.Conv3D(channels=64,
                  kernel_size=(3,1,1),
                  strides=(1,1,1),
                  padding=(1,0,0)),
        nn.BatchNorm(),
        nn.Activation(activation='relu')
    )

    (n1,n2,n3,n4) = BLOCK_CONFIG[model_depth]

    # conv_2x
    for _ in range(n1):
        net.add(R3DBlock(input_filter=64,num_filter=64,comp_index=comp_count))
        comp_count +=1

    #conv_3x
    net.add(R3DBlock(input_filter=64,num_filter=128,comp_index=comp_count,downsampling=True))
    comp_count +=1
    for _ in range(n2-1):
        net.add(R3DBlock(input_filter=128,num_filter=128,comp_index=comp_count))
        comp_count +=1
    #conv_4x
    net.add(R3DBlock(128,256,comp_index=comp_count,downsampling=True))
    comp_count+=1

    for _ in range(n3-1):
        net.add(R3DBlock(256,256))
        comp_count +=1
    #conv_5x
    net.add(R3DBlock(256,512,comp_index=comp_count,downsampling=True))
    for _ in range(n4-1):
        net.add(R3DBlock(512,512,comp_count))
        comp_count +=1
    # final layers
    net.add(nn.AvgPool3D(pool_size=(final_temporal_kernel,final_spatial_kernel,final_spatial_kernel),
                         strides=(1,1,1),
                         padding=(0,0,0)))
    net.add(nn.Dense(units=num_class))
    return net


if __name__=='__main__':
    from mxnet import init
    # net = get_R2plus1d(101,model_depth=34)
    # net.initialize()
    # print(net)
    # x = nd.random.uniform(shape=(2,3,32,112,112))
    # for layer in net:
    #     x = layer(x)
    #     print(layer.name,'output shape',x.shape)

    net2 = R2Plus2D(num_class=101,model_depth=34)
    net2.initialize(init=init.Xavier(),ctx=mx.gpu(1))
    x = nd.random.uniform(shape=(1, 3, 32, 112, 112))
    #net2.collect_params().reset_ctx(mx.gpu(1))
    x = x.as_in_context(mx.gpu(1))
    tic = time()
    for i in range(100):
        y = net2(x)
        y.wait_to_read()
    print("time consumered %f" % (time() - tic))
    #print(net2)
    #print(dir(net2))
    # x = nd.random.uniform(shape=(2, 3, 32, 112, 112))
    # net2(x)
    sym_dict = nd.load('../output/test-0017.params')
    dict_keys = sym_dict.keys()
    sym_keys = [key.split(':')[-1] for key in dict_keys]
    print(sym_keys)
    print('sym keys length',len(sym_keys))
    # #net2.load_from_sym_params('../output/test_0018.params')
    #params = net2.params
    # print(params)
    # sum_model_keys = 0
    # for i,cld in enumerate(net2._children.values()):
    #     params = cld.collect_params()
    #     net_list_name = cld.name+'_name'
    #     param_k = params.keys()
    #     print(cld.name)
    #     sum_model_keys +=len(param_k)
    #     if cld.name in ('pool0','dense0'):
    #         print(param_k)
    #         continue
    #     model_name_k = net2.__getattribute__(net_list_name)
    #     print("params length",len(param_k),"net block length",len(model_name_k))
    #     for param_name,model_name in zip(param_k,model_name_k):
    #         #print(param_name,' ',model_name)
    #         if model_name not in sym_keys:
    #             print(model_name)
    #
    # print(sum_model_keys)

    net2.load_from_sym_params('../output/test-0017.params')

    y =net2(x)
    params = net2.collect_params()

    #print(y)