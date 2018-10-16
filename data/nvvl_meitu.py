import pynvvl
from mxnet.gluon.data import Dataset,DataLoader
import cupy
import numpy as np
from mxnet import nd
import ipdb
import PIL
import os
import logging
import argparse

logger = logging.getLogger(__name__)
class MeituDataset(Dataset):
    def __init__(self,
                 data_dir,
                 label_file,
                 n_frame=32,
                 crop_size=112,
                 scale_w=171,
                 scale_h=128,
                 train=True,
                 device_ids=[0],
                 transform=None):
        super(MeituDataset,self).__init__()
        self.datadir = data_dir
        self.label_file = label_file
        self.n_frame = n_frame
        self.crop_size=crop_size
        self.scale_w = scale_w
        self.scale_h = scale_h
        self.is_train = train
        self.clip_list = []
        self.gpu_ids = device_ids # type is list
        self.load_list()

        #self.loader_list =[pynvvl.NVVLVideoLoader(device_id=temp_id, log_level='error') for temp_id in self.gpu_ids]


    def load_list(self):
        """
        load the train list to construct a file label list
        every item in self.clip_list is (file_dir,label_list)
        :return:
        """
        self.max_label = 0
        f = open(self.label_file,'r')
        for line in f.readlines():
            video_tags = line.split(',')
            video = video_tags[0]
            tags = [int(label) for label in video_tags[1:]]
            self.max_label = max(self.max_label,max(tags))
            self.clip_list.append((video,tuple(tags)))
        logger.info("load data from %s,num_clip_List, %d"%(self.label_file,len(self.clip_list)))

        # set the dataloder nvvl


    def __len__(self):
        return len(self.clip_list)

    def __getitem__(self, index):
        """
        clip a short video from video and coresponding label set
        :param index:
        :return:
        """
        temp_id = np.random.choice(self.gpu_ids, 1)[0]
        self.loader = pynvvl.NVVLVideoLoader(device_id=temp_id, log_level='info')
        #self.loader = np.random.choice(self.loader_list,1)[0]# pynvvl.NVVLVideoLoader(device_id=temp_id, log_level='error')
        video_file,tags = self.clip_list[index]
        video_file = os.path.join(self.datadir,video_file) # full path video file
        count = self.loader.frame_count(video_file)
        # start frame index
        if self.is_train:
            if count<=self.n_frame:
                frame_start=0
            else:
                frame_start =np.random.randint(0,count-self.n_frame,dtype=np.int32)
        else:
            frame_start = (count-self.n_frame)//2
        # video reshape and crop index
        if self.is_train:
            crop_x = np.random.randint(0,self.scale_w-self.crop_size,dtype=np.int32)
            crop_y = np.random.randint(0,self.scale_h-self.crop_size,dtype=np.int32)
        else:
            crop_x = (self.scale_w - self.crop_size)//2
            crop_y = (self.scale_h - self.crop_size)//2  # center crop

        video = self.loader.read_sequence(
            video_file,
            frame_start,
            count=self.n_frame,
            sample_mode='dense',
            horiz_flip=False,
            scale_height=self.scale_h,
            scale_width=self.scale_w,
            crop_y=crop_y,
            crop_x=crop_x,
            crop_height=self.crop_size,
            crop_width=self.crop_size,
            scale_method='Linear',
            normalized=False
        )
        #del self.loader
        # prepare label data
        label = np.zeros(shape=(self.max_label),dtype=np.float32)
        for tag_index in tags:
            label[tag_index]=1
        return nd.array(cupy.asnumpy(video).transpose((1,0,2,3))),nd.array(label)


def get_meitu_dataloader(data_dir,device_ids,batch_size=2,num_workers=0,n_frame=32,crop_size=112,scale_w=171,scale_h=128):
    train_label_file = os.path.join(data_dir,'DatasetLabels/short_video_trainingset_annotations.txt.082902')
    if __name__=='__main__':
        train_label_file = os.path.join(data_dir,'videos','filter.txt')
    train_dataset = MeituDataset(data_dir=os.path.join(data_dir,'videos/train_collection'),
                                 label_file= train_label_file,
                                 n_frame=n_frame,
                                 crop_size=crop_size,
                                 scale_w=scale_w,
                                 scale_h=scale_h,
                                 train=True,
                                 device_ids=device_ids)

    val_dataset = MeituDataset(data_dir=os.path.join(data_dir,'videos/val_collection'),
                               label_file=os.path.join(data_dir,'DatasetLabels/short_video_validationset_annotations.txt.0829'),
                               n_frame=n_frame,
                               crop_size=crop_size,
                               scale_w=scale_w,
                               scale_h=scale_h,
                               train=False,
                               device_ids=device_ids)
    if __name__=='__main__':
        # come to debug mode
        import time
        for i in range(0,30,3):
            data,label = train_dataset[i]
            print('data index ',i,data.shape,label.shape)
            time.sleep(0.5)
        print('finish test the same shape video')

    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
    val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers)
    return train_loader,val_loader


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='this is a dataset test parser')
    parser.add_argument('--data_dir',type=str,default='/data/jh/notebooks/hudengjun/meitu',help='this is the meitu root directory')
    args = parser.parse_args()
    train_loader,val_loader = get_meitu_dataloader(data_dir=args.data_dir,
                                                    device_ids=[0],
                                                   batch_size=2,
                                                   num_workers=2,
                                                   n_frame=32,
                                                   crop_size=112,
                                                   scale_w=171,
                                                   scale_h=128)
    for i,(batch_data,batch_label) in enumerate(train_loader):
        print('batch_data shape',batch_data.shape)
        print('batch_label shape',batch_label.shape)
        if i==2:
            break

    print("test the dataloader")

