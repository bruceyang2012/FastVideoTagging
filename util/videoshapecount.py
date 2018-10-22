import pynvvl
import os
file_name = 'meitu/DatasetLabels/short_video_trainingset_annotations.txt.082902'
shape_set = set()
data_dir = '/data/jh/notebooks/hudengjun/meitu'
with open(file_name,'r') as f:
    for line in f.readlines():
        vid_info = line.split(',')
        file_name = os.path.join(data_dir, 'videos', 'train_collection', vid_info[0])
        video_shape = pynvvl.video_size_from_file(file_name)
        shape_set.add(video_shape)
print(len(shape_set))
