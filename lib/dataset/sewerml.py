
import os

import cv2
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms  # , util
import pickle
from pdb import set_trace as stop
from PIL import Image
import json, string, sys
import torchvision.transforms.functional as TF
import random
import csv



LabelsName = ["RB","OB","PF","DE","FS","IS","RO","IN","AF","BE","FO","GR","PH","PB","OS","OP","OK"]
# VA	RB	OB	PF	DE	FS	IS	RO	IN	AF	BE	FO	GR	PH	PB	OS	OP	OK
class SewerMLDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir='/media/ubuntu/5385f1e6-0f46-4865-90c2-287b9a0f3c16/sy/Dataset_SewerML_sub_0.0625/valAll_sub_0.0625',
                 anno_path='/media/ubuntu/5385f1e6-0f46-4865-90c2-287b9a0f3c16/sy/Dataset_SewerML_sub_0.0625/valAll_sub_0.0625.csv',
                 #image_dir='/media/ubuntu/5385f1e6-0f46-4865-90c2-287b9a0f3c16/sy/complete_Sewer-ML/valAll',
                 #anno_path='/media/ubuntu/5385f1e6-0f46-4865-90c2-287b9a0f3c16/sy/complete_Sewer-ML/SewerML_Val.csv',

                 input_transform=None,
                 ):

        self.image_dir = image_dir
        print(self.image_dir)
        self.input_transform = input_transform
        self.img_names = []

        self.LabelNames = LabelsName.copy()

        gt = pd.read_csv(anno_path, sep=",", encoding="utf-8", usecols=self.LabelNames + ["Filename"])

        self.img_names = gt["Filename"].values

        self.labels = gt[self.LabelNames].values
        self.labels = np.array(self.labels).astype(int)


    def __getitem__(self, index):
        name = self.img_names[index]
        image = Image.open(os.path.join(self.image_dir, name)).convert('RGB')

        if self.input_transform:
            image = self.input_transform(image)

        labels = torch.Tensor(self.labels[index])
        return image, labels


    def __len__(self):
        return len(self.img_names)

    def GetValImageName(self):
        return self.img_names


