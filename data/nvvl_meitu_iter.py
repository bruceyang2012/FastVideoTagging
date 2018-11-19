import os
import numpy as np
import mxnet as mx
import random
import logging
import pynvvl
from lru import LRU
import multiprocessing

logging.getLogger(__name__)
class MeituClipBatchIter(mx.io.DataIter):
    def __init__(self,datadir,
                 label_file,
                 batch_size=4,
                 n_frame=32,
                 crop_size=112,
                 scale_size=128,
                 train=True,
                 lru_buffer=120,
                 gpu_id=0):
        super(MeituClipBatchIter,self).__init__(batch_size)
        self.datadir = datadir
        self.label_file = label_file
        self.batch_size = batch_size
        self.n_frame = n_frame
        self.crop_size = crop_size
        self.scale_size = scale_size
        self.train = train
        self.max_label = 0
        self.clip_lst = []
        self.load_data()
        def evicted(k,v):
            print('pop shape',k)
            del v
        self.nvvl_loader_dict = LRU(lru_buffer,evicted)
        self.gpu_id = gpu_id
        self.data_size= len(self.clip_lst)
        self.process_pool = multiprocessing.Pool(processes=10)

    def load_data(self):
        if self.train:
            video_dir = os.path.join(self.datadir,'train_collection')
        else:
            video_dir = os.path.join(self.datadir,'val_collection')
        with open(self.label_file,'r') as fin:
            for line in fin.readlines():
                vid_info = line.split(',')
                file_name = os.path.join(video_dir,vid_info[0])
                labels = [int(id) for id in vid_info[1:]]
                self.max_label = max(self.max_label,max(labels))
                self.clip_list.append((file_name,labels))
            self.max_label = self.max_label + 1
        logger.info("load data from %s,num_clip_List %d"%(video_dir,len(self.clip_list)))

    @property
    def provide_data(self):
        return [mx.io.DataDesc(name='data',shape=(self.batch_size,3,self.n_frame,self.crop_size,self.crop_size),
                               dtype=np.float32,layout='NCTHW')]

    @property
    def provide_label(self):
        return [mx.io.DataDesc(name='tags',shape=(self.batch_size,self.max_label),
                               dtype=np.float32,layout='NL')]

    def reset(self):
        self.clip_p = 0 # self.clip_p is the index to read batch data
        if self.train:
            random.shuffle(self.clip_lst)

    def next(self):
        """Get next data batch from iterator
        :return: DataBatch raise StopIteration if the end of the data is reached
        """
        # if self.clip_p<len(self.clip_lst):
        #     batch_clips = self.clip_lst[self.clip_p:min(self.data_size,self.clip_p+self.batch_size)]
        #     if len(batch_clips)<self.batch_size:
        #         batch_clips += random.sample(self.clip_lst,self.batch_size-len(batch_clips))
        #     #padding to batch_size
        #     file_names,labels = zip(*batch_clips)
        #     data = self.sample_clips(file_names)
        #     #data type is cupy,
        #     ret = mx.io.DataBatch([mx.nd.array(cupy.asnumpy(data))],[mx.nd.array(label)])
        #     self.clip_p +=self.batch_size
        #     return ret
        # else:
        #     raise StopIteration
        #Iter single video
        # def sample_clips(self,file_names):
        #     self.process_pool.map(self.decode_func,[(file_names[p], p) for p in range(len(file_names))])
        #
        # def decode_func(self,filename,p):

        if self.clip_p<self.data_size:
            filename,tags = self.clip_lst[self.clip_p]
            if(self.clip_p+1)%6==0:
                with cupy.cuda.Device(self.gpu_id):
                    cupy.get_default_memory_pool().free_all_blocks()
            video_shape = pynvvl.video_size_from_file(filename)
            loader = self.nvvl_loader_dict.get(video_shape,None)
            if loader is None:
                loader = pynvvl.NVVLVideoLoader(device_id=self.gpu_id,log_level='error')
                self.nvvl_loader_dict[video_shape]=laoder
            count = loader.frame_count(filename)
            #aug for frame start index
            if self.is_train:
                if count <= self.n_frame:
                    frame_start = 0
                else:
                    frame_start = np.random.randint(0, count - self.n_frame, dtype=np.int32)
            else:
                frame_start = (count - self.n_frame) // 2

            # rescale argumentation
            width,height = video_shape
            ow,oh = width,height
            if width<height:
                ow = self.scale_size
                oh = int(self.scale_size*height/width)
            else:
                oh = self.scale_size
                ow = int(self.scale_size*width/height)
            #random crop augu
            if self.train:
                crop_x = np.random.randint(0,ow-self.crop_size,dtype=np.int32)
                crop_y = np.random.randint(0,oh-self.crop_size,dtype=np.int32)
            else:
                crop_x = (ow - self.crop_size)//2
                crop_y = (oh - self.crop_size)//2
            video = loader.read_sequence(
                filename,
                frame_start,
                count= self.n_frame,
                sample_model=dense,
                horiz_flip=False,
                scale_height=oh,
                scale_width =ow,
                crop_y=crop_y,
                crop_x=crop_x,
                crop_height =self.crop_size,
                crop_width = self.crop_size,
                scale_method='Linear',
                normalized=False)
            labels = np.zeros(shape=(self.max_label),dtype=np.float32)
            for tag_index in tags:
                labels[tag_index]=1
            video = (video.transpose(0, 2, 3, 1) / 255 - cupy.array([0.485, 0.456, 0.406])) / cupy.array(
                [0.229, 0.224, 0.225])
            video = video.transpose(3, 0, 1, 2)
            ret = mx.io.DataBatch([mx.nd.aray(video.reshape(1,*video.shape)),],[mx.nd.array(labels),])
            self.clip_p +=1
            return ret
        else:
            raise StopIteration


def get_train_val_meitudata(data_dir,gpu_id,batch_size,n_frame,scale_size,crop_size,lru_buffer):
    train_iter = MeituClipBatchIter(datadir=data_dir,
                                    label_file=os.path.join(data_dir,'DatasetLabels/short_video_trainingset_annotations.txt.082902'),
                                    batch_size=batch_size,
                                    n_frame=n_frame,
                                    sclae_size=scale_size,
                                    crop_size=crop_size,
                                    lru_buffer=lru_buffer,
                                    gpu_id=gpu_id,
                                    train=True)
    val_iter = MeituClipBatchIter(data_dir = data_dir,
                                  label_file=os.path.join(data_dir,'DatasetLabels/short_video_validationset_annotations.txt.0829'),
                                  batch_size=batch_size,
                                  n_frame=n_frame,
                                  scale_size=scale_size,
                                  crop_size=crop_size,
                                  lru_buffer=lru_buffer,
                                  gpu_id=gpu_id,
                                  train=False)
    return train_iter,val_iter

if __name__=='__main__':
    train_iter,val_iter = get_train_val_meitudata(data_dir='/data/jh/notebooks/hudengjun/VideosFamous/FastVideoTagging/meitu',
                                                  gpu_id=0,
                                                  batch_size=4,
                                                  n_frame=32,
                                                  scale_size=126,
                                                  crop_size=112,
                                                  lru_buffer=120)
    for i,data in enumerate(train_iter):
        print(data)
        if i==2:
            break
