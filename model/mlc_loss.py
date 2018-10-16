import mxnet as mx
import mxnet
from mxnet.gluon import nn
from mxnet import nd
from mxnet import autograd
import numpy as np

class LSEP(autograd.Function):
    """
    this is the Log-sum-exp-Pairwise-ranking loss from paper:
    Improving Pairwise Ranking for Multi-label Image Classification .paper address in:
    http://ieeexplore.ieee.org/document/8099682/
    """
    name='LSEP'
    def forward(self,pred,target,max_num_trials=None):
        if max_num_trials is None:
            max_num_trials = target.shape[1]-1

        pos_mask = mxnet.nd.greater(target,0)
        neg_mask = mxnet.nd.lesser_equal(target,0)
        loss = 0
        pred_np = pred.asnumpy()
        for i in range(pred.shape[0]):
            pos = np.array([j for j,pos in enumerate(pos_mask[i]) if pos!=0]) # filter pos
            neg = np.array([j for j,neg in enumerate(neg_mask[i]) if neg!=0]) # filter neg

            for i,pj in enumerate(pos):
                for k,nj in enumerate(neg):
                    loss += np.exp(pred_np[i,nj]-pred_np[i,pj])
        loss = mxnet.nd.array([np.log(1+loss)])
        self.save_for_backward(pred,target,pos_mask,neg_mask,loss.asscalar())
        return loss

    def backward(self, grad_loss):
        pred,target,pos_mask,neg_mask,loss = self.saved_tensors
        fac = -1/loss
        grad_input = mxnet.nd.zeros_like(pred)
        ## make one-hot vecters
        one_hot_pos,one_hot_neg = [],[]

        for i in range(grad_input.shape[0]): # loop on batch each vector
            pos = np.array([j for j, pos in enumerate(pos_mask[i]) if pos != 0])  # filter pos
            neg = np.array([j for j, neg in enumerate(neg_mask[i]) if neg != 0])  # filter neg
            one_hot_pos.append(mxnet.nd.one_hot(nd.array(pos),pred.shape[1]))
            one_hot_neg.append(mxnet.nd.one_hot(nd.array(neg),pred.shape[1]))

        ## grad
        for i in range(grad_input.shape[0]): # for each pred and label sample instance
            for dum_j,phot in enumerate(one_hot_pos[i]):
                for dum_k,nhot in enumerate(one_hot_neg[i]):
                    grad_input[i] += (phot-nhot)*nd.exp(-pred[i]*(phot-nhot))
        #this is the grad input
        grad_input = grad_input*grad_loss*fac
        return grad_input,mx.nd.ones(target.shape[0], ctx=target.context)



class LsepLoss(nn.Block):
    """this is the loss function implemented :
    Imporve pairwise Ranking for multi-label image classification"""
    def __init__(self):
        super(LsepLoss,self).__init__()

    def forward(self, pred,target):
        """
        pred is the output prob,target the multi-class set label
        """
        batch,dim = pred.shape
        dist = nd.broadcast_minus(pred.reshape(batch,dim,1),pred.reshape(batch,1,dim))
        pos = mxnet.nd.greater(target,0).reshape(batch,dim,1)
        neg = mxnet.nd.equal(target,0).reshape(batch,1,dim)

        # pos_matrix = mxnet.nd.concat(*([pos]*dim),dim=2)
        # neg_matrix = mxnet.nd.concat(*([neg]*dim),dim=1)
        #print(pos_matrix.shape,neg_matrix.shape,dist.shape)
        #loss_matrix = nd.log(1+nd.sum(pos_matrix*neg_matrix*nd.exp(-dist)))
        print("----------------------")
        print("pos is ",pos)
        print("neg is ",neg)
        print("multiply is ",nd.broadcast_mul(pos,neg))
        print("the distance is ",dist)
        print("the mat mul is ",nd.broadcast_mul(pos,neg)*dist)
        print("-----------------------")
        loss_matrix = nd.log(1 + nd.sum(nd.broadcast_mul(pos,neg)* nd.exp(-dist)))
        return loss_matrix

class WarpLoss(nn.Block):
    """
    the WARP loss proposed in Wasbie and Yangqing Jia's paper
    """

    def __init__(self,label_size):
        super(WarpLoss,self).__init__()
        self.rank_weights = [1.0/1]
        for i in range(1,label_size):
            self.rank_weights.append(self.rank_weights[i-1]+1.0/(i+1))
        self.max_num_trails = label_size-1 # c-1

    def forward(self, pred,input):
        batch,dim = pred.shape
        pos = mxnet.nd.greater(target,0).reshape(batch,dim,1)
        neg = mxnet.nd.equal(target,0).reshape(batch,1,dim)

        #construct the weight Li_j
        L = mxnet.nd.zeros(shape=(batch,dim))
        for i in range(batch):
            for j in range(dim):
                if target[i,j]==1:
                    sample_score_margin = -1
                    num_trails = 0
                    while((sample_score_margin<0) and (num_trails<self.max_num_trails)):
                        neg_labels_idx = np.array([idx for idx,v in enumerate(target[i,:]) if v ==0])
                        if len(neg_labels_idx) > 0:
                            neg_idx = np.random.choice(neg_labels_idx, replace=False)
                            sample_score_margin = pred[i, neg_idx] - pred[i, j]
                            num_trails += 1
                        else:
                            num_trails=1
                            pass

                    r_j = int(np.floor(self.max_num_trails/num_trails))
                    #print("the rank of r_j is ",r_j,"the length of self.rank_weights is ",len(self.rank_weights))
                    #print(L.shape,L[i,j],self.rank_weights[r_j])
                    L[i,j] = self.rank_weights[r_j]
        #finish approximate weight

        dist = pred.reshape(batch, dim, 1) - pred.reshape(batch, 1, dim)
        pos = mxnet.nd.greater(target, 0).reshape(batch, dim, 1)
        neg = mxnet.nd.equal(target, 0).reshape(batch, 1, dim)

        # pos_matrix = mxnet.nd.concat(*([pos] * dim), dim=2)
        # neg_matrix = mxnet.nd.concat(*([neg] * dim), dim=1)
        # print(L.shape,pos_matrix.shape,neg_matrix.shape,dist.shape)
        print("L",L)
        # loss_matrix = L*nd.sum(nd.relu(1-pos_matrix*neg_matrix*dist),axis=1)
        # print(loss_matrix.shape)
        #print("weight L shape",L.shape)# namely (batch,dim)

        filter_matrix = nd.broadcast_mul(pos,neg)

        #loss_element = 1+filter_matrix*(-dist)
        loss_element = nd.relu(1+filter_matrix*(-dist))
        # print(L.shape,loss_element.shape)
        # L= nd.array([[3.0000, 5.8333, 0.0000, 0.0000],[0.0000, 3.0000, 0.0000, 3.0000]])
        L=L.reshape(batch,dim,1)
        # print("L",L)
        # print("loss_element",loss_element)
        # print(nd.broadcast_mul(L,loss_element))
        loss_matrix = nd.sum(nd.broadcast_mul(L,loss_element))
        return nd.sum(loss_matrix)


class WARP_funcLoss(autograd.Function):
    """Autograd function of WARP loss,approximate weighted rank pairwise loss"""
    name='WARP_funcLoss' # the warp loss
    def __init__(self,label_size):
        super(WARP_funcLoss, self).__init__()
        self.rank_weights=[1.0/1]
        for i in range(1,label_size):
            self.rank_weights.append(self.rank_weights[i-1]+1/(i+1))

    def forward(self,pred,target):
        batch_size = target.shape[0]
        label_size = target.shape[1]

        ## rank weight to sample and
        rank_weights = self.rank_weights
        max_num_trials = target.shape[1]-1

        pos_mask = nd.greater(target,0).asnumpy()
        neg_mask = nd.equal(target,0).asnumpy()
        L = nd.zeros_like(pred)

        for i in range(batch_size):
            for j in range(label_size):
                if target[i,j]==1:
                    ##initialization
                    sample_score_margin = -1
                    num_trials = 0
                    while ((sample_score_margin<0) and (num_trials<max_num_trials)):
                        neg_labels_idx = np.array([idx for idx,v in enumerate(target[i,:]) if v==0 ])
                        if len(neg_labels_idx) >0:
                            neg_idx = np.random.choice(neg_labels_idx,replace=False)
                            sample_score_margin = pred[i,neg_idx] - pred[i,j]
                            num_trials +=1
                        else:
                            num_trials =1
                            pass
                    ## how many trials determin the weight
                    r_j = int(np.floor(max_num_trials / num_trials))
                    L[i,j] = rank_weights[r_j]
        print("L weight",L)
        loss = nd.sum(L*(nd.sum(1 - pos_mask*pred + neg_mask*pred,axis=1,keepdims=True)),axis=1)
        self.save_for_backward(L,pos_mask,neg_mask)
        return loss

    def backward(self, grad_output):
        L,pos_mask,neg_mask = self.saved_tensors
        pos_mask = pos_mask.detach()
        neg_mask = neg_mask.detach()

        pos_grad = nd.sum(L,axis=1,keepdims=True)*(-pos_mask)
        neg_grad = nd.sum(L,axis=1,keepdims=True)*neg_mask
        grad_input = grad_output*(pos_grad+neg_grad)
        return grad_input,nd.ones_like(target.shape[0],ctx=target.context)










if __name__=='__main__':
    # lsep_loss = LsepLoss()
    # use_identity =True
    # pred = nd.random.normal(shape=(10,63))
    # target = nd.greater(nd.random.normal(shape=(10,63)),0.1)
    # if use_identity:
    #     pred= nd.array([[0.9, 0.4 ,0.5 ,0.2],[0.1 ,0.6, 0.2 ,0.8]])
    #     target = nd.array([[1,1,0,0],[0,1,0,1]])
    # pred.attach_grad()
    #
    # with autograd.record():
    #     loss = lsep_loss(pred,target)
    #     print(loss)
    #     loss.backward()
    # print(loss)
    # print(pred.grad.shape)
    # print(pred,target)
    # print("pred grad",pred.grad)
    #
    #
    # print("--------------- the warp loss")
    #
    # if use_identity:
    #     pred = nd.array([[0.9, 0.4, 0.5, 0.2], [0.1, 0.6, 0.2, 0.8]])
    #     target = nd.array([[1, 1, 0, 0], [0, 1, 0, 1]])
    # pred.attach_grad()
    # warp_loss = WarpLoss(label_size=4)
    # with autograd.record():
    #     loss = warp_loss(pred,target)
    #     print("warp loss",loss)
    #     loss.backward()
    #     print(pred.grad.shape)
    #     print(pred, target)
    #     print(pred.grad)


    warp_loss = WarpLoss(label_size=63)
    pred = nd.random.normal(shape=(10,63))
    target = nd.random.normal(shape=(10,63))
    target = nd.greater(target,0.2)
    use_identity=True
    if use_identity:
        pred = nd.array([[0.9, 0.4, 0.5, 0.2], [0.1, 0.6, 0.2, 0.8]])
        target = nd.array([[1, 1, 0, 0], [0, 1, 0, 1]])
    #print("target",target)
    pred.attach_grad()
    with autograd.record():
        loss = warp_loss(pred,target)
        print(loss)
        loss.backward()
        print("pred.grad",pred.grad)


    # test for Lsep loss function implementaion
    Lsep_funcLoss = LSEP()
    if use_identity:
        pred = nd.array([[0.9, 0.4, 0.5, 0.2], [0.1, 0.6, 0.2, 0.8]])
        target = nd.array([[1, 1, 0, 0], [0, 1, 0, 1]])
    pred.attach_grad()
    with autograd.record():
        loss = Lsep_funcLoss(pred,target)
        loss.backward(mxnet.nd.ones_like(loss))
        print("loss value",loss.as_scalar())
    print(pred.grad)

    # test for autograd function edition of WARPLoss

