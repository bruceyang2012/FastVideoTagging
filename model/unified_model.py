# the unified model is cnn rnn model,the cnn model extract video and image feature, the rnn model the label relationship
from mxnet.gluon import nn
from model import R2Plus2D
import mxnet.gluon.rnn as rnn
from mxnet import nd
import mxnet

class Encoder(nn.Block):
    # the video clip encoder model part
    def __init__(self,num_class,model_depth,embed_size):
        super(Encoder,self).__init__()
        self.basic_model = R2Plus2D(num_class=num_class,model_depth=model_depth)
        self.linear = nn.Dense(units=embed_size)
        self.bn = nn.BatchNorm(axis=1,momentum=0.1)

    def forward(self, x):
        r2_feature = self.basic_model.extract_features(x)
        r2_feature = r2_feature.reshape(r2_feature.shape[0],-1) # batch,-1
        features = self.linear(r2_feature)
        features = self.bn(features)
        return features


class Decoder(nn.Block):
    # the video decoder to label set part
    def __init__(self,embed_size,hidden_size,vocab_size,num_layers,max_seq_length=20):
        super(Decoder,self).__init__()
        self.embed = nn.Embedding(vocab_size,embed_size)
        self.lstm = rnn.GRU(hidden_size=hidden_size,num_layers=1,dropout=0.1)

    def forward(self, features):
        pass

if __name__=='__main__':
    encoder = Encoder(num_class=62,model_depth=34,embed_size=100)
    encoder.initialize(mxnet.init.Xavier())
    video_clip = nd.random.normal(shape=(4,3,32,112,112))
    feature = encoder(video_clip)
    print("feature shape",feature.shape)