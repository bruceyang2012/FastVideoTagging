import nvvl
# this is a pytorch nvvl ucf-101 dataloader
import torch
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms.transforms as T
import os

class UCF101_NVVL():
    def __init__(self,data_dir,
                 n_frames=32,
                 is_cropped=True,
                 crop_size=112,
                 device_id=0,
                 fp16=False):
        self.data_dir = data_dir
        self.n_frames = n_frames
        self.is_cropped = is_cropped
        self.crop_size = crop_size
        self.device_id = device_id
        self.fp16 = fp16

