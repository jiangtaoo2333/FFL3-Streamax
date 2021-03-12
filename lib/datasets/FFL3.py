# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# ------------------------------------------------------------------------------

import os
import random
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
import torch.utils.data as data
import pandas as pd
from PIL import Image
import math
import numpy as np

from ..utils.transforms import fliplr_joints, crop, generate_target, transform_pixel, getlist
from ..utils.transforms import getpoints,pts21to19

class FFL3_Dataset(data.Dataset):

    def __init__(self, cfg, is_train=True, transform=None):

        # specify annotation file for dataset
        if is_train:
            self.fileList = getlist(cfg.DATASET.TRAINSET,'.jpg')
        else:
            self.fileList = getlist(cfg.DATASET.TESTSET,'.jpg')

        self.ImgList = []
        print('len(fileList):',len(self.fileList))

        for imgPath in self.fileList:
            ptsPath = imgPath.replace('.jpg','.txt').replace('/picture_mask/','/landmark/')
            try:
                getpoints(ptsPath)
                self.ImgList.append(imgPath)
            except:
                continue

        print('len(ImgList):',len(self.ImgList))

        self.is_train = is_train
        self.transform = transform
        self.data_root = cfg.DATASET.ROOT
        self.input_size = cfg.MODEL.IMAGE_SIZE
        self.output_size = cfg.MODEL.HEATMAP_SIZE
        self.sigma = cfg.MODEL.SIGMA
        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rot_factor = cfg.DATASET.ROT_FACTOR
        self.label_type = cfg.MODEL.TARGET_TYPE
        self.flip = cfg.DATASET.FLIP

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.ImgList)

    def __getitem__(self, idx):

        # get image and pts
        if(1):
            image_path = self.ImgList[idx]
            img = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)
            pts_path = image_path.replace('.jpg','.txt').replace('/picture_mask/','/landmark/')
            pts = np.genfromtxt(pts_path,skip_header=1)
            pts = pts.astype('float').reshape(-1, 2)
            assert pts.shape[0] == 106

            minX,minY = pts.min(axis=0)
            maxX,maxY = pts.max(axis=0)

        # get centor and scale
        if(1):
            center_w = (math.floor(minX) + math.ceil(maxX)) / 2.0
            center_h = (math.floor(minY) + math.ceil(maxY)) / 2.0

            scale = max(math.ceil(maxX) - math.floor(minX), math.ceil(maxY) - math.floor(minY)) / 200.0
            center = torch.Tensor([center_w, center_h])

        scale *= 1.25
        nparts = pts.shape[0]

        r = 0
        if self.is_train:

            scale = scale * (random.uniform(1 - self.scale_factor,
                                            1 + self.scale_factor))

            if random.random() <= 0.6:
                r = random.uniform(-self.rot_factor, self.rot_factor)
            else:
                r = 0

            if random.random() <= 0.5 and self.flip:
                img = np.fliplr(img)
                pts = fliplr_joints(pts, width=img.shape[1], dataset='FFL3')
                center[0] = img.shape[1] - center[0]

        img = crop(img, center, scale, self.input_size, rot=r)

        target = np.zeros((nparts, self.output_size[0], self.output_size[1]))
        tpts = pts.copy()

        for i in range(nparts):
            if tpts[i, 1] > 0:
                tpts[i, 0:2] = transform_pixel(tpts[i, 0:2]+1, center,
                                               scale, self.output_size, rot=r)
                target[i] = generate_target(target[i], tpts[i]-1, self.sigma,
                                            label_type=self.label_type)
        img = img.astype(np.float32)
        img = (img/255.0 - self.mean) / self.std
        img = img.transpose([2, 0, 1])
        target = torch.Tensor(target)
        tpts = torch.Tensor(tpts)
        center = torch.Tensor(center)

        meta = {'index': idx, 'center': center, 'scale': scale,
                'pts': torch.Tensor(pts), 'tpts': tpts}

        return img, target, meta


if __name__ == '__main__':

    print('a')