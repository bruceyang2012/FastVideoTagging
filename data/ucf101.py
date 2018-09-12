import mxnet as mx
from mxnet.gluon.data import Dataset,DataLoader
import mxnet.gluon.data.vision.transforms as T
import os
import csv
import cv2
import numpy as np
import random
import logging
from mxnet import nd
from mxnet.image import imdecode
import ipdb
import PIL

logger = logging.getLogger(__name__)

class UCF101(Dataset):
    def __init__(self,datadir,n_frame=32,crop_size=112,
                 scale_w=171,scale_h=128,train=True,transform=None):
        super(UCF101,self).__init__()
        self.datadir = datadir
        self.n_frame = n_frame
        self.crop_size = crop_size
        self.scale_w = scale_w
        self.scale_h = scale_h
        self.is_train = train
        self.clip_lst = []
        self._transform = transform
        self.load_list()


    def load_list(self):
        """load all the video informatin in the file list"""
        id2class_name = {}
        class_names = []
        with open(os.path.join(self.datadir, 'classInd.txt')) as fin:
            for i, nm in csv.reader(fin, delimiter=' '):
                id2class_name[int(i) - 1] = nm
            for i in range(len(id2class_name)):
                class_names.append(id2class_name[i])

        if self.is_train:
            with open(os.path.join(self.datadir, 'trainlist01.txt')) as fin:
                for nm, c in csv.reader(fin, delimiter=' '):
                    self.clip_lst.append((os.path.join(self.datadir, nm), int(c) - 1))
        else:
            with open(os.path.join(self.datadir, 'testlist01.txt')) as fin:
                for nm, in csv.reader(fin, delimiter=' '):
                    c = nm[:nm.find('/')]
                    self.clip_lst.append((os.path.join(self.datadir, nm), class_names.index(c)))
        # loging
        logger.info("load data from %s,num_clip_List %d"%(self.datadir,len(self.clip_lst)))

    def __len__(self):
        return len(self.clip_lst)

    def __getitem__(self, index):
        """the index is the video index in clip_list,read several frame from the index"""
        filename,label = self.clip_lst[index]
        if not os.path.exists(filename):
            print("the file not exist",filename)
            return None
        cthw_data = None
        nd_image_list = []
        while len(nd_image_list) is 0:
            v = cv2.VideoCapture(filename)
            width = v.get(cv2.CAP_PROP_FRAME_WIDTH)
            height= v.get(cv2.CAP_PROP_FRAME_HEIGHT)
            length = v.get(cv2.CAP_PROP_FRAME_COUNT)

            assert self.crop_size<=width and self.crop_size<= height,'%d'
            length = int(length)
            if length<self.n_frame:
                logger.info("%s length %d <%d"%(filename,length,self.n_frame))
                # the following operation will tail the last frame

            # set the sample begin frame id
            if not self.is_train:
                frame_st = 0 if length<=self.n_frame else int((length-self.n_frame)//2)
            else:
                frame_st = 0 if length<=self.n_frame else random.randrange(length-self.n_frame+1)

            # set random crop position in single frame
            if self.is_train:
                row_st = random.randrange(self.scale_h - self.crop_size + 1)
                col_st = random.randrange(self.scale_w - self.crop_size + 1)
            else:
                row_st = int((self.scale_h - self.crop_size) / 2)
                col_st = int((self.scale_w - self.crop_size) / 2)

            # allocate the capacity to store image and jump to the position

            v.set(cv2.CAP_PROP_POS_FRAMES,frame_st)
            #start to read the following frames by current start position
            import ipdb
            #ipdb.set_trace()
            for frame_p in range(min(self.n_frame,length)):
                _,f = v.read()
                if f is not None:
                    f = cv2.resize(f,(self.scale_w,self.scale_h))  #in dim of hwc
                    f = cv2.cvtColor(f,cv2.COLOR_BGR2RGB)
                    f=f[row_st:row_st + self.crop_size, col_st:col_st + self.crop_size, :]
                    if self._transform:
                        nd_image_list.append(self._transform(nd.array(f))) # the frame_p transform

                else:
                    nd_image_list.clear() #clear the image_list
                    break
        # after transform return CHW dim
        # replication the last frame if the length < self.n_frame
        current_length = len(nd_image_list)
        cthw_data = nd.stack(*nd_image_list,axis=1) #from CHW, to CTHW
        #tmp = nd.zeros(shape=(self.n_frame, self.crop_size, self.crop_size, 3), dtype='float32')
        if current_length<self.n_frame:
            #construct the last frame and concat
            extra_data = nd.tile(nd_image_list[-1],reps=(self.n_frame-current_length,1,1,1))
            extra_data = extra_data.transpose((1,0,2,3))
            cthw_data = nd.concat(cthw_data,extra_data,dim=1)
        return cthw_data,label

train_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

def get_ucf101trainval(datadir,
                       batch_size=4,
                       n_frame=32,
                       crop_size=112,
                       scale_h=128,
                       scale_w=171,
                       num_workers=6):
    """construct the dataset and then set the datasetloader"""
    train_dataset = UCF101(datadir,n_frame,crop_size,scale_w,scale_h,train=True,transform=train_transform)
    val_dataset = UCF101(datadir,n_frame,crop_size,scale_w,scale_h,train = False,transform=train_transform)
    if __name__=='__main__':
        # test get data with single video
        data0 = train_dataset[0]
        print(data0)
        val_data = val_dataset[12]
        print(val_data)
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=6)
    val_dataloader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=6)
    return train_dataloader,val_dataloader

if __name__=='__main__':

    # train_loader,val_loader = get_ucf101trainval()
    # for i,data in train_loader:
    #     print(type(data))
    #     ipdb.set_trace()
    #     break
    train_data = UCF101(datadir='/data/jh/notebooks/hudengjun/DeepVideo/UCF-101',n_frame=32,
                       crop_size=112,
                       scale_h=128,
                       scale_w=171,
                       train=True,
                        transform=train_transform)
    data = train_data[0]
    print(data)

