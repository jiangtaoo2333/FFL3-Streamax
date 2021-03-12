'''
@Author: Jiangtao
@Date: 2020-03-30 17:00:40
@LastEditors: Jiangtao
@LastEditTime: 2020-04-08 09:23:34
@Description: 
'''
import os
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import sys
import cv2

import lib.models as models
# from lib.config.HrNet_defaults import _C as config
from lib.config.DSM_defaults import _C as config
from lib.config import update_config
from PIL import Image
import numpy as np
from lib.utils.transforms import crop
from lib.core.evaluation import decode_preds,NormRmse
from lib.core.function import getlist
from lib.utils import utils
import time
import uuid
from tqdm import tqdm
from xml.dom.minidom import Document
import xml.dom.minidom

GPU_ID = utils.NoOfGpu
device = torch.device('cuda:{}'.format(GPU_ID))

def parse_args():
    parser = argparse.ArgumentParser(description='Train Face Alignment')

    parser.add_argument('--cfg',
                        default='./experiments/300w/face_alignment_300w_hrnet_w18.yaml',
                        help='experiment configuration filename', type=str)

    parser.add_argument('--model-file', help='model parameters',
                        default='./models/HR18-DSM-old.pth', type=str)

    parser.add_argument("--aim", type=str, default="all", choices=["eye", 'mouth', 'all'])

    args = parser.parse_args()
    update_config(config, args)
    return args

getted_args = parse_args()

def calculat_Eye_Error(pts_pre,pts_gt):

    norm = np.linalg.norm(pts_gt[1:5, :].mean(axis=0).reshape((1,2)) - pts_gt[6:10, :].mean(axis=0).reshape(1,2), axis=1)

    pts_gt = pts_gt[0:11]

    loss_eye = np.linalg.norm(pts_pre-pts_gt,axis=1,keepdims=True).mean()

    norm = norm[0]

    loss = loss_eye / norm

    return norm,loss

def calculat_all_Error(pts_pre,pts_gt):
    
    norm = np.linalg.norm(pts_gt[1:5, :].mean(axis=0).reshape((1,2)) - pts_gt[6:10, :].mean(axis=0).reshape(1,2), axis=1)

    loss = np.linalg.norm(pts_pre-pts_gt,axis=1,keepdims=True).mean()

    loss = loss / norm

    return norm,loss

def getbox(xmlFile):

    # xmlFile = imgFile.replace('/img','/xml')
    xmlFile = os.path.splitext(xmlFile)[0] + '.xml'

    if not os.path.exists(xmlFile):
        return np.zeros((1,4))
        
    dom = xml.dom.minidom.parse(xmlFile)  
    root = dom.documentElement

    itemlist = root.getElementsByTagName('xmin')
    minX = int(float(itemlist[0].firstChild.data))

    itemlist = root.getElementsByTagName('ymin')
    minY = int(float(itemlist[0].firstChild.data))

    itemlist = root.getElementsByTagName('xmax')
    maxX = int(float(itemlist[0].firstChild.data))

    itemlist = root.getElementsByTagName('ymax')
    maxY = int(float(itemlist[0].firstChild.data))

    boxes = np.zeros((1,4))

    boxes[0][0] = minX
    boxes[0][1] = minY
    boxes[0][2] = maxX
    boxes[0][3] = maxY

    return boxes
    
def prepare_input(image, bbox, image_size):
    """

    :param image:The path to the image to be detected
    :param bbox:The bbox of target face
    :param image_size: refers to config file
    :return:
    """
    scale = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200
    center_w = (bbox[0] + bbox[2]) / 2
    center_h = (bbox[1] + bbox[3]) / 2
    center = torch.Tensor([center_w, center_h])
    scale *= 1.25
    img = np.array(Image.open(image).convert('RGB'), dtype=np.float32)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = crop(img, center, scale, image_size, rot=0)
    img = img.astype(np.float32)
    img = (img / 255.0 - mean) / std
    img = img.transpose([2, 0, 1])
    img = torch.Tensor(img)
    img = img.unsqueeze(0)
    return img, center, scale

class faceAlignment():
    """ This is a custom engine for this training cycle """

    def __init__(self, args=getted_args):    

        self.args = args 
        if(0):
            self.model = models.get_face_alignment_net(config)
            state_dict = torch.load(args.model_file,map_location=device)
            new_state_dict =  {k.replace('module.',''):v for k,v in state_dict.items()}
            self.model.load_state_dict(new_state_dict)
        if(0):
            self.model = torch.load(args.model_file)

        if(1):
            self.model = models.get_face_alignment_net(config)
            self.oldmodel = torch.load(args.model_file,map_location='cuda:0')
            oldmodel_dict = self.oldmodel.state_dict()
            self.model.load_state_dict(oldmodel_dict)
            torch.save(self.model, './models/HR18-DSM-old.pth',_use_new_zipfile_serialization=False)

        self.model.eval()
        self.model.cuda(GPU_ID)

    def align(self,img,box):


        imagepath = './a.jpg'
        cv2.imwrite(imagepath,img)

        inp, center, scale = prepare_input(imagepath, box, config.MODEL.IMAGE_SIZE)
        inp = inp.cuda(GPU_ID)
        # t1 = time.time()
        output = self.model(inp)
        # t2 = time.time()
        # print(t2 - t1)
        score_map = output.data.cpu()
        center = center.reshape(1,2)
        scale = [scale]
        # t3 = time.time()
        preds = decode_preds(score_map, center, scale, [64, 64])
        preds = preds.numpy()
        preds = preds.reshape((-1,2))
        # t4 = time.time()
        # print(t4 - t3)
        
        return preds

if __name__ == '__main__':

    faceAlignment = faceAlignment()
    imgFile = './images/a.jpg'
    img = cv2.imread(imgFile,1)

    pre = faceAlignment.align(img,[206,384,561,800])
    pts_pre = pre
    eye = pts_pre[0:11,]
    mouth = pts_pre[11:,]

    image = cv2.imread(imgFile,-1)
    cv2.rectangle(image,(int(206),int(384)),(int(561),int(800)),(0,125,0))
    for point in pre:
        cv2.circle(image, (int(point[0]),int(point[1])), 2, (255, 255, 0), 1)
    cv2.imwrite(imgFile.replace('.jpg','_hrnet.jpg'),image)







