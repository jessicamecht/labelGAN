import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import PIL
from PIL import Image,ImageOps
from torchvision import transforms
from torch.utils.data import Dataset
import pickle
import csv

class ChestXrayDatasetSegmentation(Dataset):

    def __init__(self, root_dir, image_paths, label_paths, use_aug=True):

        self.image_paths = image_paths
        self.label_paths = label_paths
        self.root_dir = root_dir
        self.use_aug = use_aug

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
 
        #Reading images
        img_name = os.path.join(self.root_dir, 'original/images', self.image_paths[idx])
        image = Image.open(img_name)
        
        #Converting to grayscale if RGB
        image = ImageOps.grayscale(image)
        
        #Reading Labels
        label_name = os.path.join(self.root_dir, 'original/masks', self.label_paths[idx])
        label = Image.open(label_name)
        
        #Converting to tensors and Resizing images
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))])
        self.transform_hflip = transforms.functional.hflip
        
        image = self.transform(image)
        label = self.transform(label)
        
        #Flipping images and labels with probability 0.5
        if self.use_aug:
            probability = torch.rand(1)
            if probability <= 0.5:
                image = self.transform_hflip(image)
                label = self.transform_hflip(label)

        return image,label
    
class AugmentationDatasetSegmentation(Dataset):

    def __init__(self, image_paths, labels_path, path_file):
        
        self.image_paths = image_paths
        self.label_paths = labels_path
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
 
        #Reading images
        img_name = os.path.join(self.root_dir, 'synthetic/images', self.image_paths[idx])
        image = Image.open(img_name)
        
        #Converting to grayscale if RGB
        image = ImageOps.grayscale(image)
        
        #Reading Labels
        label_name = os.path.join(self.root_dir, 'synthetic/masks', self.label_paths[idx])
        label = Image.open(label_name)
        label = np.where(np.min(label, axis=2) >= 150, 1, 0)
        
        #Converting to tensors and Resizing images
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))])
        self.transform_hflip = transforms.functional.hflip
        
        image = self.transform(image)
        label = self.transform(label)
        
        #Flipping images and labels with probability 0.5
        probability = torch.rand(1)
        if probability <= 0.5:
            image = self.transform_hflip(image)
            label = self.transform_hflip(label)

        return image, label
    

def get_datasets(train_size, use_augmentation = False):
    root_dir = 'data/segmentation'
    
    # file containing shuffled file paths (input and labels) for train/test/val sets
    with open('segmentation_metadata.pkl', 'rb') as f:
        segmentation_metadata = pickle.load(f)
    
    train_dataset = ChestXrayDatasetSegmentation(root_dir,
                                                 metadata['train_image_paths_original'][0:train_size],
                                                 metadata['train_label_paths_original'][0:train_size]) 
    valid_dataset = ChestXrayDatasetSegmentation(root_dir,
                                                 metadata['valid_image_paths'],
                                                 metadata['valid_label_paths'])   
    test_dataset = ChestXrayDatasetSegmentation(root_dir,
                                                metadata['test_image_paths'],
                                                metadata['test_label_paths'],
                                                use_aug=False)  
    
    if use_augmentation:
        augmentation_dataset = AugmentationDatasetSegmentation(root_dir,
                                                               metadata['train_image_paths_synthetic'][0:train_size],
                                                               metadata['train_label_paths_synthetic'][0:train_size]) 
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, augmentation_dataset])
    
    return train_dataset, valid_dataset, test_dataset

