import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os

class MovingMNISTDataset(Dataset):
    def __init__(self, train, transform=None, target_transform=None, folder=""):
        if train:
            filename = os.path.join(folder,"movingmnisttrain.npz")
        else:
            filename = os.path.join(folder,"movingmnisttest.npz")
        assert os.path.isfile(filename)
	#N x C x W x H -> N x W x H x C 
        self.X = np.load(filename)['arr_0'].transpose(0,3,2,1)
        self.transform = transform
        self.target_transform = target_transform
        
    def __getitem__(self, index):
        img   = self.X[index]
        label = self.X[index]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
            
        return img, label
    def __len__(self):
        return len(self.X)
