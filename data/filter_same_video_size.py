#import pynvvl
import os
import cv2
import pynvvl
datadir = '/data/jh/notebooks/hudengjun/meitu/'
shape_file_dir = os.path.join(datadir,'videos')
from data.extract_key_frame import extract_key_frame
def filter_same_size(data_dir,shape_file_dir):
    """
    to filter the same size video
    :param data_dir:
    :param shape_file_dir:
    :return:
    """
    base_train_video_dir = os.path.join(datadir,'videos','train_collection')
    file_list = os.path.join(datadir,'DatasetLabels','short_video_trainingset_annotations.txt.082902')
    f =open(file_list,'r')
    basic_info = f.readline()
    video_fname = basic_info.split(',')[0]
    rec = cv2.VideoCapture(os.path.join(base_train_video_dir,video_fname))
    ret, frame = rec.read()
    shape = frame.shape[:2]
    f_out = open(os.path.join(shape_file_dir,'filter.txt'),'w')
    count =0
    for line in f.readlines():
        video_fname = line.split(',')[0]
        rec = cv2.VideoCapture(os.path.join(base_train_video_dir,video_fname))
        ret,frame = rec.read()
        if ret:
            if frame.shape[:2]==shape:
                f_out.write(line)
                print(video_fname)
                f_out.flush()
                count +=1
                if count==100:
                    break
    f_out.close()
    f.close()

import json
def filter_video_size(datadir):
    base_train_video_dir = os.path.join(datadir, 'videos', 'train_collection')
    file_list = os.path.join(datadir, 'DatasetLabels', 'short_video_trainingset_annotations.txt.082902')
    shape_count = {}
    with open(file_list, 'r') as f:
        for line in f.readlines():
            video_fname = line.split(',')[0]
            shape = pynvvl.video_size_from_file(os.path.join(base_train_video_dir,video_fname))
            count = shape_count.get(shape,0)
            shape_count[shape]=count+1
    with open("shape_count.json",'w') as fp:
        json.dump(shape_count,fp)
        print("finish shape dump to json file")
if __name__=='__main__':
    #filter_same_size(datadir,shape_file_dir)
    filter_video_size(datadir)