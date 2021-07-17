from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np
import torch
import torchvision.transforms as T
import cv2


class ukiyoe2photo(Dataset):
    def __init__(self, root_uki = "data/ukiyoe2photo/trainA", root_pic = "data/ukiyoe2photo/trainB" , transformU = None, transformP = None, **kwargs):
    
        if transformU is None:
            self.transformU = lambda x: x
        else:
            self.transformU = transformU
        if transformP is None:
            self.transformP = lambda x: x
        else:
            self.transformP = transformP

        self.uki_images = os.listdir(root_uki)
        self.pic_images = os.listdir(root_pic)
        self.length_dataset = max(len(self.uki_images), len(self.pic_images))
        self.uki_len = len(self.uki_images)
        self.pic_len = len(self.pic_images)

        u = [cv2.imread(os.path.join(root_uki, pa)) for pa in self.uki_images]
        p = [cv2.imread(os.path.join(root_pic, pa)) for pa in self.pic_images]
    
        self.u = np.asarray(u) 
        self.p = np.asarray(p)
    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        uki_img = self.u[index % self.uki_len]
        pic_img = self.p[index % self.pic_len]

        return self.transformU(uki_img), self.transformP(pic_img)

def test():
    dataset = ukiyoe2photo()
    print(dataset.u.shape)
    print(dataset.p.shape)

    for channel in (0,1,2):
        ch = dataset.u[:,:,:,channel] / 255
        print(channel, ch.mean(), ch.std())
        ch = dataset.p[:,:,:,channel] / 255
        print(channel, ch.mean(), ch.std())

    


if __name__ == "__main__":
    test()
    
