import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import PIL
from PIL import Image,ImageOps
from torchvision import transforms
from torch.utils.data import Dataset, ConcatDataset
import pickle

class ChestXrayDatasetClassification(Dataset):

    def __init__(self, images_dir, image_and_labels, basic_transform_aug = None):

        self.images_dir = images_dir
        self.image_and_labels = image_and_labels
        self.transform_aug = basic_transform_aug

    def __len__(self):
        return len(self.image_and_labels)

    def __getitem__(self, idx):
         
        image_path, labels = self.image_and_labels[idx]
        
        #Reading images
        img_name = os.path.join(self.images_dir, image_path)
        image = Image.open(img_name)
        
        #Converting to grayscale if RGB
        image = ImageOps.grayscale(image)
        
        if self.transform_aug is not None:
            image = self.transform_aug(image)
        
        #Converting to tensors and Resizing images
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))])
        self.transform_hflip = transforms.functional.hflip
        
        image = self.transform(image)
                
        return image, torch.from_numpy(labels)
    
class AugmentationDatasetClassification(Dataset):

    def __init__(self, images_dir, image_and_labels):
        
        self.images_dir = images_dir
        self.image_and_labels = image_and_labels
    
    def __len__(self):
        return len(self.image_and_labels)

    def __getitem__(self, idx):
        
        image_path, labels = self.image_and_labels[idx]
 
        #Reading images
        img_name = os.path.join(self.images_dir, image_path)
        image = Image.open(img_name)
        
        #Converting to grayscale if RGB
        image = ImageOps.grayscale(image)
        
        #Converting to tensors and Resizing images
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))])
        self.transform_hflip = transforms.functional.hflip
        
        image = self.transform(image)

        return image, torch.from_numpy(labels)
    

def get_data_splits(f):
    return [(image.split()[0], np.array(list(map(int, image.split()[1:])))) for image in f.readlines()]

def get_train_data(images_dir,
                   use_augmentation = False,
                   augmentation_type = "basic",
                   augmentation_size = 50):
    with open("./vinbig/train_binarized_list.txt") as f:
        image_and_labels = get_data_splits(f)
        
    train_dataset = ChestXrayDatasetClassification(images_dir,
                                                   image_and_labels) 
    
    if use_augmentation:
        if augmentation_type == "basic":
            basic_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(0.4),
                transforms.RandomAffine(degrees=20, scale=(1.1, 1.1)),
                transforms.RandomCrop((28, 28), padding=2, pad_if_needed=True, fill=0,
                                      padding_mode='constant'), transforms.ToTensor()
            ])
            augmentation_dataset = ChestXrayDatasetClassification(images_dir,
                                                                  image_and_labels[:augmentation_size],
                                                                  basic_transform)
    
        elif augmentation_type == "random":
            with open("./vinbig/random_binarized_list.txt") as f:
                image_and_labels = get_data_splits(f)
            augmentation_dataset = AugmentationDatasetClassification(images_dir,
                                                                     image_and_labels[:augmentation_size]) 
        elif augmentation_type == "kde":
            with open("./vinbig/kde_binarized_list.txt") as f:
                image_and_labels = get_data_splits(f)
            augmentation_dataset = AugmentationDatasetClassification(images_dir,
                                                                     image_and_labels[:augmentation_size]) 
            
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, augmentation_dataset])
    
    return train_dataset
        
def get_val_data(images_dir):
    with open("./vinbig/val_binarized_list.txt") as f:
        image_and_labels = get_data_splits(f)
    
    val_dataset = ChestXrayDatasetClassification(images_dir,
                                                 image_and_labels) 
    
    return val_dataset

def get_test_data(images_dir):
    with open("./vinbig/test_binarized_list.txt") as f:
        image_and_labels = get_data_splits(f)
    
    test_dataset = ChestXrayDatasetClassification(images_dir,
                                                 image_and_labels) 
    
    return test_dataset

