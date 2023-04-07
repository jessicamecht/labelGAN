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

class ChestXrayDataset(Dataset):

    def __init__(self, root_dir, image_and_labels, use_aug=False):

        self.image_and_labels = image_and_labels
        self.root_dir = root_dir
        self.use_aug = use_aug

    def __len__(self):
        return len(self.image_and_labels)

    def __getitem__(self, idx):
 
        #Reading images
        img_name = os.path.join(self.root_dir, 'originals', f"{self.image_and_labels[idx][0]}_json", "img.png")
        image = Image.open(img_name)
        
        #Converting to grayscale if RGB
        image = ImageOps.grayscale(image)
        
        #Reading segmentation Masks
        mask_name = os.path.join(self.root_dir, 'originals', f"{self.image_and_labels[idx][0]}_json", "label.png")
        mask = Image.open(mask_name)
        mask = ImageOps.grayscale(mask)
        
        #Extracting disease label
        label = self.image_and_labels[idx][1]
        
        #Converting to tensors and Resizing images
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))])
        self.transform_hflip = transforms.functional.hflip
        
        image = self.transform(image)
        mask = self.transform(mask)
        
        data = {
                "cxr": image,
                "gaze": mask, # named gaze with compatibility with MT-UNet code
                "Y": label
        }
        
        return data
    
class AugmentationDataset(Dataset):

    def __init__(self, image_paths, mask_paths, labels):
        
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.labels = labels
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
 
        #Reading images
        img_name = os.path.join(self.root_dir, 'synthetic/images', self.image_paths[idx])
        image = Image.open(img_name)
        
        #Converting to grayscale if RGB
        image = ImageOps.grayscale(image)
        
        #Reading segmentation Masks
        mask_name = os.path.join(self.root_dir, 'synthetic/masks', self.mask_paths[idx])
        mask = Image.open(mask_name)
        mask = np.where(np.min(mask, axis=2) >= 150, 1, 0)
        
        #Extracting disease label
        label = self.labels[idx]
        
        #Converting to tensors and Resizing images
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))])
        self.transform_hflip = transforms.functional.hflip
        
        image = self.transform(image)
        mask = self.transform(mask)
        
        #Flipping images and labels with probability 0.5
        probability = torch.rand(1)
        if probability <= 0.5:
            image = self.transform_hflip(image)
            mask = self.transform_hflip(mask)

        data = {
                "cxr": image,
                "gaze": mask, # named gaze with compatibility with MT-UNet code
                "Y": label
        }
        
        return data

def get_data_splits(f):
    return [(image.split()[0], np.array(list(map(int, image.split()[1:])))) for image in f.readlines()]

def get_datasets(aug_size=None, use_augmentation = False):
    root_dir = '../../../../data/vinbig_we_labeled/'
    
    with open(os.path.join(root_dir, "train_binarized_list.txt")) as f:
        train_file = get_data_splits(f)
    train_dataset = ChestXrayDataset(root_dir,
                                     train_file) 
    
    with open(os.path.join(root_dir, "train_binarized_list.txt")) as f:
        val_file = get_data_splits(f)
    valid_dataset = ChestXrayDataset(root_dir,
                                     val_file)   
    
    with open(os.path.join(root_dir, "train_binarized_list.txt")) as f:
        test_file = get_data_splits(f)
    test_dataset = ChestXrayDataset(root_dir,
                                    test_file)  
    
    if use_augmentation:
        synthetic_labels = "To-do: obtain disease labels for synthetic images"
        augmentation_dataset = AugmentationDataset(root_dir,
                                                   os.listdir(os.path.join(self.root_dir, 'synthetic/images'))[0:aug_size],
                                                   os.listdir(os.path.join(self.root_dir, 'synthetic/masks'))[0:aug_size],
                                                   synthetic_labels) 
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, augmentation_dataset])
    
    return train_dataset, valid_dataset, test_dataset

