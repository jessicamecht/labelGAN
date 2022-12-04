import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import PIL
from PIL import Image,ImageOps
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
import pickle
import csv

with open('paths.pkl', 'rb') as f:
    paths = pickle.load(f)
    
root_dir = '../../dataset_xray/'

class ChestXrayDataset(Dataset):

    def __init__(self,root_dir,image_paths,label_paths,use_aug=True):

        self.image_paths = image_paths
        self.label_paths = label_paths
        self.root_dir = root_dir
        self.use_aug = use_aug

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
 
        #Reading images
        img_name = os.path.join(self.root_dir,'images',self.image_paths[idx])
        image = Image.open(img_name)
        
        #Converting to grayscale if RGB
        image = ImageOps.grayscale(image)
        
        #Reading Labels
        label_name = os.path.join(self.root_dir,'masks',self.label_paths[idx])
        label = Image.open(label_name)
        
        #Converting to tensors and Resizing images
        self.transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((256,256))])
        self.transform_hflip = transforms.functional.hflip
        
        image = self.transform(image)
        label = self.transform(label)
        
        #Flipping images and labels with probability 0.5
        if self.use_aug:
            probability = torch.rand(1)
            if probability<=0.5:
                image = self.transform_hflip(image)
                label = self.transform_hflip(label)

        return image,label
    
class AugmentationDataset(Dataset):

    def __init__(self,path_file):
        
        self.root_dir = '../../vis_2000/'
       
        with open(path_file, mode ='r') as file:
            csvFile = csv.reader(file)
            self.image_paths = [line[1] for line in csvFile][1:]
        self.label_paths = [i.replace('image','mask') for i in self.image_paths]
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
 
        #Reading images
        img_name = os.path.join(self.root_dir,self.image_paths[idx])
        image = Image.open(img_name)
        
        #Converting to grayscale if RGB
        image = ImageOps.grayscale(image)
        
        #Reading Labels
        label_name = os.path.join(self.root_dir,self.label_paths[idx])
        label = Image.open(label_name)
        label = np.where(np.min(label,axis=2)>=150,1,0)
        
        #Converting to tensors and Resizing images
        self.transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((256,256))])
        self.transform_hflip = transforms.functional.hflip
        
        image = self.transform(image)
        label = self.transform(label)
        
        #Flipping images and labels with probability 0.5
        probability = torch.rand(1)
        if probability<=0.5:
            image = self.transform_hflip(image)
            label = self.transform_hflip(label)

        return image,label
    
class BaselineAugmentationDataset(Dataset):

    def __init__(self):
        
        self.root_dir = '../../semanticGan_generations/'
       
        image_ids = [i.replace('img','') for i in os.listdir(self.root_dir) if i.startswith('img')]
        label_ids = [i.replace('mask','') for i in os.listdir(self.root_dir) if i.startswith('mask')]
        pair_ids = list(set(image_ids).intersection(set(label_ids)))
        self.image_paths = [('img'+i) for i in pair_ids]
        self.label_paths = [('mask'+i) for i in pair_ids]
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
 
        #Reading images
        img_name = os.path.join(self.root_dir,self.image_paths[idx])
        image = Image.open(img_name)
        
        #Converting to grayscale if RGB
        image = ImageOps.grayscale(image)
        
        #Reading Labels
        label_name = os.path.join(self.root_dir,self.label_paths[idx])
        label = Image.open(label_name)
        label = np.where(np.min(label,axis=2)>0,1,0)
        
        #Converting to tensors and Resizing images
        self.transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((256,256))])
        self.transform_hflip = transforms.functional.hflip
        
        image = self.transform(image)
        label = self.transform(label)
        
        #Flipping images and labels with probability 0.5
        probability = torch.rand(1)
        if probability<=0.5:
            image = self.transform_hflip(image)
            label = self.transform_hflip(label)

        return image,label
    

def get_datasets(train_size,use_augmentation = False):
    train_dataset =  ChestXrayDataset(root_dir,paths['train_image_paths'][0:train_size],
                                      paths['train_label_paths'][0:train_size]) 
    valid_dataset =  ChestXrayDataset(root_dir,paths['valid_image_paths'],paths['valid_label_paths'])   
    test_dataset =  ChestXrayDataset(root_dir,paths['test_image_paths'],paths['test_label_paths'],use_aug=False)  
    
    if use_augmentation=='baseline':
        augmentation_dataset = BaselineAugmentationDataset() 
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, augmentation_dataset])
        
    elif use_augmentation:
        augmentation_dataset = AugmentationDataset('../../outliers_new.csv') 
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, augmentation_dataset])
    return train_dataset,valid_dataset,test_dataset

'''
dataset = ChestXrayDataset(root_dir,image_paths,label_paths)
train_size = int(0.1 * len(dataset))
valid_size = int(0.1 * len(dataset))
test_size = len(dataset)-train_size-valid_size
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, 
                                                                           [train_size, valid_size, test_size])
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
'''
