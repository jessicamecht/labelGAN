import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import PIL
from PIL import Image,ImageOps
from torchvision import transforms
from torch.utils.data import Dataset
import pickle

class ChestXrayDatasetClassification(Dataset):

    def __init__(self, root_dir, image_paths, labels, use_aug=True):

        self.image_paths = image_paths
        self.labels = labels
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
        
        #Extracting label
        label = self.labels[idx]
        
        #Converting to tensors and Resizing images
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))])
        self.transform_hflip = transforms.functional.hflip
        
        image = self.transform(image)
        
        #Flipping images and labels with probability 0.5
        if self.use_aug:
            probability = torch.rand(1)
            if probability <= 0.5:
                image = self.transform_hflip(image)

        return image, label
    
class AugmentationDatasetClassification(Dataset):

    def __init__(self, image_paths, labels, path_file):
        
        self.image_paths = image_paths
        self.labels = labels
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
 
        #Reading images
        img_name = os.path.join(self.root_dir, 'synthetic/images', self.image_paths[idx])
        image = Image.open(img_name)
        
        #Converting to grayscale if RGB
        image = ImageOps.grayscale(image)
        
        #Extracting label
        label = self.labels[idx]
        
        #Converting to tensors and Resizing images
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))])
        self.transform_hflip = transforms.functional.hflip
        
        image = self.transform(image)
        
        #Flipping images and labels with probability 0.5
        probability = torch.rand(1)
        if probability <= 0.5:
            image = self.transform_hflip(image)

        return image, label
    

def get_datasets(train_size, use_augmentation = False):
    root_dir = 'data/classification'
    
    # file containing shuffled file paths (inputs only) and labels for train/test/val sets
    with open('classification_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    train_dataset = ChestXrayDatasetClassification(root_dir,
                                                   metadata['train_image_paths_original'][0:train_size],
                                                   metadata['tmetadatarain_labels_original'][0:train_size]) 
    valid_dataset = ChestXrayDatasetClassification(root_dir,
                                                   metadata['valid_image_paths'],
                                                   metadata['valid_labels'])   
    test_dataset = ChestXrayDatasetClassification(root_dir,
                                                  metadata['test_image_paths'],
                                                  metadata['test_labels'],
                                                  use_aug=False)  
    
    if use_augmentation:
        augmentation_dataset = AugmentationDatasetSegmentation(root_dir,
                                                               metadata['train_image_paths_synthetic'][0:train_size],
                                                               metadata['train_labels_synthetic'][0:train_size]) 
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, augmentation_dataset])
    
    return train_dataset, valid_dataset, test_dataset

