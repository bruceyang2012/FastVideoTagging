# the unified model is cnn rnn model,the cnn model extract video and image feature, the rnn model the label relationship
from mxnet.gluon import nn
from model import R2Plus2D
import mxnet.gluon.rnn as rnn
from mxnet import nd
import mxnet
from mxnet.gluon.model_zoo.vision import get_resnet
class Encoder(nn.Block):
    # the video clip encoder model part
    def __init__(self,num_class,model_depth):
        super(Encoder,self).__init__()
        self.basic_model = R2Plus2D(num_class=num_class,model_depth=model_depth)


    def forward(self, x):
        # x shape is NCTHW,just use the conv to avg_pool feature extract
        r2_feature = self.basic_model.extract_features(x)
        return r2_feature
    def custom_load_params(self,filename):
        self.basic_model.load_parameters(filename)

class State_Trans(nn.Block):
    # input is the cnn out dim ,output is the rnn hidden dim
    def __init__(self,input_dim,output_dim):
        super(State_Trans,self).__init__():
        with self.name_scope():
            self.linear1 = nn.Dense(output_dim,in_units=input_dim,flatten=False)

    def forward(self, feat):
        return self.linear1(feat)

class Decoder(nn.Block):
    # the video decoder to label set part
    def __init__(self,embed_size,hidden_size,vocab_size,num_layers,max_seq_length=20):
        super(Decoder,self).__init__()
        self.num_layers = num_layers
        self.vid_init_state = nn.Dense(hidden_size,flatten=False)
        self.embed = nn.Embedding(vocab_size,embed_size)
        self.rnn = rnn.GRU(hidden_size=hidden_size,num_layers=num_layers,dropout=0.1)

    def forward(self, inputs,states):
        embeded = self.embed(inputs)
        outputs,states = self.rnn(embeded,states)
        return outputs,states

    def begin_state(self,*args,**kwargs):
        #useage decoder.begin_state(batch_size=4,func=nd.zeros,vid_feat = features)
        video_feat = kwargs['vid_feat']
        init_state = self.vid_init_state(video_feat)#
        init_state = init_state.reshape(1,*(init_state.shape)) # LNC for layer is 1
        kwargs.pop('vid_feat')
        states = self.rnn.begin_state(*args,**kwargs)
        states[0] = nd.broadcast_axis(init_state,size=self.num_layers,axis=0)
        return states