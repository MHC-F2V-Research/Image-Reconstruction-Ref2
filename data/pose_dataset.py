import os.path
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
import numpy as np
import pickle
import random

class PoseDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        if (self.opt.isTrain):
            phase = 'train'
        else:
            phase = 'test'
        self.root = os.path.join(self.opt.dataroot, phase)
        self.transform=transforms.Compose([transforms.ToTensor()])

    def randomFlip(self, x, x_target, pose, pose_target, mask, mask_target):
        # random horizontal flip
        rand = random.randint(0, 1)
        if (rand==1):
            x = np.flip(x, axis=1).copy()
            x_target = np.flip(x_target, axis=1).copy()
            pose = np.flip(pose, axis=1).copy()
            pose_target = np.flip(pose_target, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
            mask_target = np.flip(mask_target, axis=1).copy()
        return x, x_target, pose, pose_target, mask, mask_target

    def __getitem__(self, index):
        # CHANGE PATH BC ITS NOT OPENING THE CORRECT DATASET
        # loop for maybe opening both jpgs and pngs? not sure if this will work
        # don't forget to import image and some other things first before implementing this one?
        # imgs = []
        # valid_images = [".jpg",".png"]
        # for f in os.listdir(path):
        #     ext = os.path.splitext(f)[1]
        #     if ext.lower() not in valid_images:
        #         continue
        #         imgs.append(Image.open(os.path.join(path,f)))
        path = os.path.join(self.root,'%d.jpg'%index) #all the img from robot are png? change this later
        pickle_in = open(path,"rb")
        sample = pickle.load(pickle_in)

        x = sample['x']                     #(256, 256, 3)
        x_target = sample['x_target']       #(256, 256, 3)
        pose = sample['pose']               #(256, 256, 18)
        pose_target = sample['pose_target'] #(256, 256, 18)
        mask = sample['mask']               #(256, 256, 1)
        mask_target = sample['mask_target'] #(256, 256, 1)

        # random fliping
        if (self.opt.isTrain):
            x, x_target, pose, pose_target, mask, mask_target = self.randomFlip(x, x_target, pose, pose_target, mask, mask_target)

        # to tensor
        x = self.transform(x)
        x_target = self.transform(x_target)
        pose = self.transform(pose)
        pose_target = self.transform(pose_target)
        mask = self.transform(mask)
        mask_target = self.transform(mask_target)

        # input-guide-target
        input = (x/255)*2-1
        guide = pose_target
        target = (x_target/255)*2-1
        # Put data into [input, guide, target]
        return {'A':input, 'guide':guide, 'B':target}


    def __len__(self):
        if (self.opt.isTrain):
            return 73340
        else:
            return 12800

    def name(self):
        return 'PoseDataset'
