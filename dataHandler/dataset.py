from typing import Any
import torch
from torch.utils.data import Dataset
import numpy as np
import os

class MHDProjDataset(Dataset):
    def __init__(self,root_dir,transforms=None,random_rotate=True,sorted=False,test_batch_length=None, test_file_list=None):
        super().__init__()
        self.dir=root_dir
        self.transforms=transforms
        self.file_list=os.listdir(root_dir)
        self.random_rotate=random_rotate
        if sorted:
            self.file_list.sort()
        self.length=len(self.file_list)
        self.test_batch_length=test_batch_length
        if test_file_list is None:
            self.train_file_list=self.file_list[:self.__len__()]
            self.test_file_list=self.file_list[self.__len__():]
        else:
            self.test_file_list=test_file_list
            self.train_file_list=[file for file in self.file_list if not(file in self.test_file_list)]
            self.file_list = self.train_file_list + self.test_file_list ## So that training datapoints come first
    def __len__(self):
        if not(self.test_batch_length is None):
            return self.length-self.test_batch_length
        else:
            return self.length
    def __getitem__(self, index):
        np_img=np.load(os.path.join(self.dir,self.file_list[index]))
        if self.transforms:
            if self.random_rotate:
                np_img=np.rot90(np_img,np.random.randint(0,4),axes = (-1, -2))
            tens=torch.from_numpy(np_img.copy())
            return self.transforms(tens)
        else:
            if self.random_rotate:
                np_img=np.rot90(np_img,np.random.randint(0,4),axes = (-1, -2))
            tens=torch.from_numpy(np_img.copy())
            return tens
    def test_batch(self):
        if self.test_batch_length is None:
            raise ValueError('No test batch length provided at init -> no test batch created')
        np_img_list=[]
        if self.random_rotate:
            for i in range(self.test_batch_length):
                np_img_list.append(np.rot90(np.load(os.path.join(self.dir,self.test_file_list[i])),np.random.randint(0,4),axes = (-1, -2)))
        else:
            for i in range(self.test_batch_length):
                np_img_list.append(np.load(os.path.join(self.dir,self.test_file_list[i])))
        np_batch=np.stack(np_img_list,axis=0) 
        batch_tensor=torch.from_numpy(np_batch)
        if len(batch_tensor.shape)==3:
            batch_tensor=batch_tensor.unsqueeze(1) ###to comply with torch conventions of having B x C x H x W
        if self.transforms:
            return self.transforms(batch_tensor)
        else:
            return batch_tensor


class LogNormalTransform(object):
    def __init__(self,mu=5.5,sigma=0.3):
        self.mu=mu
        self.sigma=sigma
    def __call__(self,img):
        return((torch.log(img)-self.mu)/self.sigma)
    #def __repr__(self):
    #    return "Custom LogNormal transformation, applying a log and then reducing to get a standard normal distribution"


class ReTransform(object):
    def __init__(self, mu=0, sigma=1) -> None:
        self.mu=mu
        self.sigma=sigma
    def __call__(self, img) -> Any:
        return (torch.real(img) - self.mu)/self.sigma