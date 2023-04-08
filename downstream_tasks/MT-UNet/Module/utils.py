import torch 
import numpy as np 

def dice_coeff(pred, target):
    smooth = 1.
    pred = torch.where(pred>=0.5, 1, 0)
    iflat = pred.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))

def get_atleast_one_metric(pred, true):
    pred = pred > 0.5
    return torch.any(torch.logical_and(pred, true)).long().item()

def binary_class_eval(Y_class, Y_class_pred):
    Y_class = Y_class > 0.5
    Y_class_pred = Y_class_pred > 0.5
    nofinding = Y_class[:, 8] 
    nofinding_pred = Y_class_pred[:, 8] 
    disease = (Y_class[:, 0:8].sum(dim=1) + Y_class[:, 9:].sum(dim=1)) > 0
    disease_pred = (Y_class_pred[:, 0:8].sum(dim=1) + Y_class_pred[:, 9:].sum(dim=1)) > 0
    return (nofinding == nofinding_pred) | (disease == disease_pred)

def pixel_acc(Y_seg_pred, Y_seg):
    batch_size = Y_seg_pred.shape[0]
    num_pixels = batch_size * Y_seg_pred.shape[2] * Y_seg_pred.shape[3]
    num_correct_pixels = (Y_seg_pred == Y_seg).sum()
    return num_correct_pixels / num_pixels