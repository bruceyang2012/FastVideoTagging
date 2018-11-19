import mxnet.gluon.nn as nn
import mxnet as mx

class Decision_thresh(nn.Block):
    def __init__(self,thresh_size=63):
        super(Decision_thresh,self).__init__()
        self.thresh = self.params.get(
                'thresh', init=mx.init.Constant(0),
                shape=(1, thresh_size))

    def forward(self,x):
        """ x shape is N*63 ,self.threshold is 1*63"""
        x = x-self.thresh.data()
        return x

class Decision_topk(nn.Block):
    def __init__(self,confidence_C=63,K_way=4):
        self.linear1 = nn.Dense(in_units=confidence_C,units=(confidence_C+K_way)//2,\
                                use_bias=True,activation='relu')
        self.linear2 = nn.Dense(units=K_way)

    def forward(self,x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x # x shape is N*K_way,to pred top_k is the output_label.loss is SoftmaxwithCrossentropy


if __name__=='__main__':
    from mxnet.gluon.loss import  SigmoidBinaryCrossEntropyLoss
    from mxnet import nd,autograd
    model = Decision_thresh(thresh_size=4)
    model.initialize(init=mx.init.Xavier())
    x = nd.array([[0.1,0.7,0.9,0.4],[0.8,0.5,0.8,0.1]])
    label = nd.array([[0,1,1,0],[1,0,0,0]])
    loss_criterion = SigmoidBinaryCrossEntropyLoss()
    with autograd.record():
        y_pred = model(x)
        loss = loss_criterion(y_pred,label)
        print("loss",nd.sum(loss).asscalar())
        loss.backward()
        print(model.thresh.grad())

    # to test the Decision_topk model to predict the top_k is groud truth

    model2 = Decision_topk(confidence_C=63,K_way=4)
    mdoel2.initialize(init=mx.init.Xavier())
    #x = nd.

