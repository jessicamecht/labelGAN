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

    def __init__(self, images_dir, images, aug_transform=None):
        
        self.images_dir = images_dir
        self.images = images
        self.aug_transform = aug_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
 
        #Reading images
        img = self.images[idx].strip()
        img_name = os.path.join(self.images_dir, 'original/images', img)
        image = Image.open(img_name)
        
        #Converting to grayscale if RGB
        image = ImageOps.grayscale(image)
        
        #Reading Labels
        mask = img.replace(".png", "") + "_mask.png" if "CHN" in img else img
        label_name = os.path.join(self.images_dir, 'original/masks', mask)
        label = Image.open(label_name)
        
        if self.aug_transform is not None:
            image = self.aug_transform(image)
            label = self.aug_transform(label)
        
        #Converting to tensors and Resizing images
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))])
        self.transform_hflip = transforms.functional.hflip
        
        image = self.transform(image)
        label = self.transform(label)

        return image, label
    
class AugmentationDatasetSegmentation(Dataset):

    def __init__(self, images_dir, images, aug_type):
        
        self.images_dir = images_dir
        self.images = images
        self.aug_type = aug_type
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
 
        #Reading images
        img = self.images[idx].strip()
        img_name = os.path.join(self.images_dir, f'{self.aug_type}/images', img)
        image = Image.open(img_name)
        
        #Converting to grayscale if RGB
        image = ImageOps.grayscale(image)
        
        #Reading Labels
        label_name = os.path.join(self.images_dir, f'{self.aug_type}/masks', img)
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
    

    
def get_train_data(images_dir,
                   train_size = 100,
                   use_augmentation = False,
                   augmentation_type = "basic",
                   augmentation_size = 50,
                   augmentation_dir = ""):
    """
    - The {set}_list.txt contains only image name and segregated into original, random, kde directories.
    - Each of these directories is further segregated into images and masks dirs for input and label respectively.
    
    - All aforementioned sets should be available in the images_dir directory,
    and the set split file names should be placed in shenken repo directory.
    """
    
    with open("./shenken/train_list.txt") as f:
        images = f.readlines()
        
    train_dataset = ChestXrayDatasetSegmentation(images_dir,
                                                 images[:train_size]) 
    
    if use_augmentation:
        if augmentation_type == "basic":
            basic_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomAffine(degrees=20, scale=(1.1, 1.1)),
                transforms.RandomCrop((28, 28), padding=2, pad_if_needed=True, fill=0,
                                      padding_mode='constant'), transforms.ToTensor()
            ])
            augmentation_dataset = ChestXrayDatasetSegmentation(images_dir,
                                                                images[:augmentation_size],
                                                                basic_transform)
    
        elif augmentation_type == "random":
            with open("./shenken/random_list.txt") as f:
                images = f.readlines()
            augmentation_dataset = AugmentationDatasetSegmentation(images_dir,
                                                                   images[:augmentation_size]) 
        elif augmentation_type == "kde":
            with open("./shenken/kde_list.txt") as f:
                images = f.readlines()
            augmentation_dataset = AugmentationDatasetSegmentation(images_dir,
                                                                   images[:augmentation_size]) 
            
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, augmentation_dataset])
    
    return train_dataset
        
def get_val_data(images_dir):
    with open("./shenken/val_list.txt") as f:
        images = f.readlines()
    
    val_dataset = ChestXrayDatasetSegmentation(images_dir,
                                               images) 
    
    return val_dataset

def get_test_data(images_dir):
    with open("./shenken/test_list.txt") as f:
        images = f.readlines()
    
    test_dataset = ChestXrayDatasetSegmentation(images_dir,
                                                images) 
    
    return test_dataset

    
# def get_datasets(train_size, use_augmentation = False):
#     root_dir = 'data/segmentation'
    
#     # file containing shuffled file paths (input and labels) for train/test/val sets
#     with open('segmentation_metadata.pkl', 'rb') as f:
#         segmentation_metadata = pickle.load(f)
    
#     train_dataset = ChestXrayDatasetSegmentation(root_dir,
#                                                  metadata['train_image_paths_original'][0:train_size],
#                                                  metadata['train_label_paths_original'][0:train_size]) 
#     valid_dataset = ChestXrayDatasetSegmentation(root_dir,
#                                                  metadata['valid_image_paths'],
#                                                  metadata['valid_label_paths'])   
#     test_dataset = ChestXrayDatasetSegmentation(root_dir,
#                                                 metadata['test_image_paths'],
#                                                 metadata['test_label_paths'],
#                                                 use_aug=False)  
    
#     if use_augmentation:
#         augmentation_dataset = AugmentationDatasetSegmentation(root_dir,
#                                                                metadata['train_image_paths_synthetic'][0:train_size],
#                                                                metadata['train_label_paths_synthetic'][0:train_size]) 
#         train_dataset = torch.utils.data.ConcatDataset([train_dataset, augmentation_dataset])
    
#     return train_dataset, valid_dataset, test_dataset

