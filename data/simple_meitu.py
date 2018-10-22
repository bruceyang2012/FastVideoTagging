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



# datadir = '/data/jh/notebooks/hudengjun/meitu'
# video_data = '/data/jh/notebooks/hudengjun/meitu/videos/train_collection'
# val_data = '/data/jh/notebooks/hudengjun/meitu/videos/val_collection'
# train_label = '/data/jh/notebooks/hudengjun/meitu/DatasetLabels/shor-xxxx.txt'

class SimpleMeitu(Dataset):
    def __init__(self,datadir,n_frame=32,crop_size=112,
                 scale_w=171,scale_h=128,train=True,transform=None):
        super(SimpleMeitu,self).__init__()
        self.datadir = datadir
        self.n_frame = n_frame
        self.crop_size = crop_size
        self.scale_w = scale_w
        self.scale_h = scale_h
        self.is_train = train
        self.clip_lst = []
        self._transform = transform
        self.max_label =0
        self.load_list()


    def load_list(self):
        """load all the video informatin in the file list"""
        if self.is_train:
            label_file = os.path.join(self.datadir, 'DatasetLabels','short_video_trainingset_annotations.txt.082902')
            folder_name = 'train_collection'
        else:
            label_file = os.path.join(self.datadir, 'DatasetLabels', 'short_video_validationset_annotations.txt.0829')
            folder_name = 'val_collection'

        with open(label_file,'r') as fin:
            for line in fin.readlines():
                vid_info = line.split(',')
                file_name = os.path.join(self.datadir,'videos',folder_name,vid_info[0])
                labels = [int(id) for id in vid_info[1:]]
                self.max_label = max(self.max_label,max(labels))
                self.clip_lst.append((file_name,labels))
        logger.info("load data from %s,num_clip_List %d"%(self.datadir,len(self.clip_lst)))

    def __len__(self):
        return len(self.clip_lst)

    def __getitem__(self, index):
        """the index is the video index in clip_list,read several frame from the index"""
        filename,labels = self.clip_lst[index]
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

            #assert self.crop_size<=width and self.crop_size<= height
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
        # begin to construct the label
        label_nd = np.zeros(shape=(self.max_label), dtype=np.float32)
        for tag_index in labels:
            label_nd[tag_index-1] = 1
        return cthw_data,label_nd

train_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

def get_simple_meitu_dataloader(datadir,
                       batch_size=4,
                       n_frame=32,
                       crop_size=112,
                       scale_h=128,
                       scale_w=171,
                       num_workers=6):
    """construct the dataset and then set the datasetloader"""
    train_dataset = SimpleMeitu(datadir,n_frame,crop_size,scale_w,scale_h,train=True,transform=train_transform)
    val_dataset = SimpleMeitu(datadir,n_frame,crop_size,scale_w,scale_h,train = False,transform=train_transform)
    if __name__=='__main__':
        # test get data with single video
        data0 = train_dataset[0]
        print("train data nframe shape",data0[0].shape,"train data labels ",data0[1])
        val_data = val_dataset[12]
        print("val data nframe shape",val_data[0].shape,"val data labels",val_data[1])

    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=6)
    val_dataloader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=6)
    return train_dataloader,val_dataloader

if __name__=='__main__':

    train_data = SimpleMeitu(datadir='/data/jh/notebooks/hudengjun/meitu',n_frame=32,
                       crop_size=112,
                       scale_h=128,
                       scale_w=171,
                       train=True,
                        transform=train_transform)
    data = train_data[0]


    train_loader,val_loader = get_simple_meitu_dataloader(datadir='/data/jh/notebooks/hudengjun/meitu',n_frame=32,crop_size=112,
                                                          scale_h=128,scale_w=171,num_workers=6)
    for i,(data,label) in enumerate(train_loader):
        print(data.shape)
        print(label.shape)

