import sys
import json
from tqdm import tqdm

import datasets_custom
from dice_loss import DiceLoss
from hzhu_MTL_UNet import *

import time

import torch
from torch import nn as nn
import os
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryJaccardIndex
from torchmetrics import Dice

# torch.backends.cudnn.enabled=False

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import argparse

# create argument parser object
parser = argparse.ArgumentParser(description='')

# add arguments
parser.add_argument('-gpu_num', '--gpu_num', type=int, required=False, help='')
parser.add_argument('-use_augment', '--use_augment', action=argparse.BooleanOptionalAction)
parser.add_argument('-num_epochs', '--num_epochs', type=int, required=False, default=50, help='')
parser.add_argument('-train_bs', '--train_bs', type=int, required=False, default=8, help='')
parser.add_argument('-val_bs', '--val_bs', type=int, required=False, default=2, help='')
parser.add_argument('-test_bs', '--test_bs', type=int, required=False, default=1, help='')
parser.add_argument('-aug_size', '--aug_size', type=int, required=False, default=None, help='')
parser.add_argument('-aug_types', '--aug_types', type=str, nargs="*", required=False, default=[None], help='')
parser.add_argument('-resize_px', '--resize_px', type=int, required=False, default=512, help='')
parser.add_argument('-val_check', '--val_check', type=int, required=False, default=5, help='')
parser.add_argument('-lr', '--lr', type=float, required=False, default=1e-3, help='')
parser.add_argument('-beta1', '--beta1', type=float, required=False, default=0.5, help='')

parser.add_argument('-mode', '--mode', type=str, required=True, default="train", help='')
parser.add_argument('-model_path', '--model_path', type=str, required=False, default="", help='')

"""
usage for no augmentation === python MT_unet-rahul.py -gpu_num 0 -train_bs 1 -resize_px 1024 -val_check 5 -num_epochs 100
basic augmentation === python MT_unet-rahul.py -gpu_num 0 -train_bs 2 -resize_px 1024 -val_check 1 -num_epochs 100 -use_augment -aug_type basic -aug_size 50
"""

# parse arguments from command line
args = parser.parse_args()

# use arguments
use_augmentation = args.use_augment
augmentation_types = args.aug_types
aug_size = args.aug_size
gpu_num = args.gpu_num
num_epochs = args.num_epochs
train_bs = args.train_bs
test_bs = args.test_bs
val_bs = args.val_bs
resize_px = args.resize_px
val_check = args.val_check
lr = args.lr
beta1 = args.beta1
mode = args.mode


augment_type_str = '_'.join(augmentation_types) if use_augmentation else "None"
save_path = f"/data1/shared/jessica/drive_data/results/aug_type_{augment_type_str}_aug_size_{aug_size}_use_augment_{use_augmentation}_num_epochs_{num_epochs}_resize_px_{resize_px}/"
model_path = args.model_path if args.model_path != "" else save_path + "best_model.pt"

if not os.path.exists(save_path):
    os.makedirs(save_path)

train_dataset, val_dataset, test_dataset = datasets_custom.get_datasets(res=(resize_px, resize_px),
                                                                        aug_size = aug_size,
                                                                        use_augmentation = use_augmentation,
                                                                        aug_types = augmentation_types)

dataAll = {
            "Train": DataLoader(train_dataset,
                                batch_size=train_bs,
                                shuffle=True),
            "Valid": DataLoader(val_dataset, batch_size=val_bs, shuffle=True),
            "Test": DataLoader(test_dataset,
                               batch_size=test_bs,
                               shuffle=True) 
}

device = torch.device(f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu")
model = MTL_UNet_Main(in_channels = 1, out_dict = {'class': 15, 'image': 1},)
model = model.to(device)

classification_loss = nn.BCELoss()
seg_pred_loss = DiceLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, 0.999))

#Function for visualization
def visualize(cxr, mask, Y, seg_pred, disease_pred, epoch):
    batchsize = cxr.size()[0]
    fig, axs = plt.subplots(batchsize, 3, figsize=(10, 10))
    
    if batchsize > 1:
        for i in range(batchsize): 
            img_plot = cxr[i].permute(1,2,0).detach().cpu()
            seg_pred_plot = seg_pred[i].permute(1,2,0).detach().cpu()
            label_plot = mask[i].permute(1,2,0).detach().cpu()

            axs[i, 0].imshow(img_plot)
            axs[i, 0].set_title('Image')

            axs[i, 1].imshow(seg_pred_plot, cmap='gray')
            axs[i, 1].set_title('Prediction')

            axs[i, 2].imshow(label_plot, cmap='gray')
            axs[i, 2].set_title('Label')
    else:
        img_plot = cxr[0].permute(1,2,0).detach().cpu()
        seg_pred_plot = seg_pred[0].permute(1,2,0).detach().cpu()
        label_plot = mask[0].permute(1,2,0).detach().cpu()

        axs[0].imshow(img_plot)
        axs[0].set_title('Image')

        axs[1].imshow(seg_pred_plot, cmap='gray')
        axs[1].set_title('Prediction')

        axs[2].imshow(label_plot, cmap='gray')
        axs[2].set_title('Label')
    
    plt.savefig(f"{save_path}/{epoch}.png")
    fig.clf()
    
    return

#Dice Coefficient
def dice_coeff(pred, target):
    pred = (pred > 0.5).long()
    dice = Dice()
    
    return dice(pred, target)

# disease v/s no-disease binary classifier
def get_binary_acc(pred, target):
    target = target.squeeze(0)
    pred = pred.squeeze(0)
    pred = pred > 0.5
    
    if target[8].item() == 1:
        if pred[8].long().item() == 1: 
            return 1
        else: 
            return 0
    else:
        return torch.any(torch.logical_and(pred, target)).long().item()

# pixel-acc
def get_pixel_acc(pred, target):
    pred = pred > 0.5
    return (torch.logical_and(pred, target).long().sum() / target.sum()).item()

# at least one classified metric
def get_atleast_one_metric(pred, target):
    pred = pred > 0.5
    return torch.any(torch.logical_and(pred, target)).long().item()

#Validation Loop
def validation(model):
    losses = []
    model.eval()
    for cxr, mask, Y in dataAll["Valid"]:
        X = cxr.to(device)
        seg = mask.type(torch.int8).to(device)
        Y = Y.to(device)
        
        Y_pred, seg_pred = model(X)

        loss = model.compute_loss(
                    y_class_pred = Y_pred.type(torch.float),
                    y_image_pred = seg_pred.type(torch.float),
                    y_class_true = Y.type(torch.float), 
                    y_image_true = seg.type(torch.float),
                    loss_class = classification_loss,
                    loss_image = seg_pred_loss)
        
        losses.append(loss.item())
    return np.mean(losses)

def prec_rec(predictions, targets):
    predictions = predictions > 0.5
    accuracy = (predictions == targets).float().mean().item()
    # Calculate the precision, recall, and F1 score
    tp = (predictions & targets).sum(dim=1)
    fp = (predictions & ~targets).sum(dim=1)
    fn = (~predictions & targets).sum(dim=1)
    precision = (tp / (tp + fp + 1e-7)).mean().item()
    recall = (tp / (tp + fn + 1e-7)).mean().item()
    f1 = (2 * precision * recall) / (precision + recall + 1e-7)
    return accuracy, f1


# test model
def test(model, tset = "Test"):
    dice_scores = []
    iou = []
    atleast_one_score = []
    pixel_acc = []
    binary_acc = []
    losses = []
    jaccard = BinaryJaccardIndex()
    accs = []
    f1s = []
    
    model.eval()
    with torch.no_grad():
        for cxr, mask, Y in dataAll[tset]:
            X = cxr.to(device)
            seg = mask.type(torch.int8).to(device)
            Y = Y.to(device)

            Y_pred, seg_pred = model(X)
            
            loss = model.compute_loss(
                    y_class_pred = Y_pred.type(torch.float),
                    y_image_pred = seg_pred.type(torch.float),
                    y_class_true = Y.type(torch.float), 
                    y_image_true = seg.type(torch.float),
                    loss_class = classification_loss,
                    loss_image = seg_pred_loss)
        
            losses.append(loss.item())
            
            iou.append(jaccard(seg_pred.cpu(), seg.cpu()))
            dice_scores.append(dice_coeff(seg_pred.cpu(), seg.cpu()))
            atleast_one_score.append(get_atleast_one_metric(Y_pred, Y))
            pixel_acc.append(get_pixel_acc(seg_pred.cpu(), seg.cpu()))
            binary_acc.append(get_binary_acc(Y_pred, Y))
            acc, f1 = prec_rec(Y_pred, Y)
            accs.append(acc)
            f1s.append(f1)
    
    return  (np.mean(iou),
             np.mean(dice_scores),
             np.mean(atleast_one_score),
             np.mean(pixel_acc),
             np.mean(binary_acc),
             np.mean(losses),
             np.mean(accs),
             np.mean(f1s))

# train model
def train():
    loss_list = []
    best_loss = np.inf
    for epoch in tqdm(range(num_epochs)):
#         start = time.process_time()
        model.train()
#         print(time.process_time() - start)
        for cxr, mask, Y in dataAll["Train"]: 
            X = cxr.to(device).type(torch.float)

            Y = Y.to(device).type(torch.float)
            seg = mask.to(device).type(torch.int8)
            
            optimizer.zero_grad()
            Y_pred, seg_pred = model(X)

            loss = model.compute_loss(
                    y_class_pred = Y_pred.type(torch.float),
                    y_image_pred = seg_pred.type(torch.float),
                    y_class_true = Y.type(torch.float), 
                    y_image_true = seg.type(torch.float),
                    loss_class = classification_loss,
                    loss_image = seg_pred_loss)

    #         loss = seg_pred_loss(Y_seg_pred.type(torch.float), Y_seg.type(torch.float)) + classification_loss(Y_class_pred.type(torch.float), Y_class.type(torch.float))
    
            loss.backward()
            optimizer.step()
            loss_list.append(loss.detach().clone().cpu())
        
        visualize(cxr, mask, Y, seg_pred, Y_pred, epoch)

        print("[TRAIN] Epoch : %d, Loss : %2.5f" % (epoch, np.mean(loss_list)))       
        if (epoch + 1) % val_check == 0:
#             iou, dsc, custom_acc, pixel_acc, binary_acc, cur_loss = test(model)
#             print("[TEST] Epoch : %d, IoU: %2.3f, DSC: %2.3f, aACC: %.3f, pACC: %3.3f, bACC: %3.3f" % (epoch, iou, dsc, custom_acc, pixel_acc, binary_acc))
            cur_loss = validation(model)

            print("[VALIDATION] Epoch : %d, Loss : %2.5f" % (epoch, cur_loss))
            if cur_loss < best_loss:
                best_loss = cur_loss
                torch.save(model.state_dict(), os.path.join(save_path, f"best_model.pt")) 
        torch.save(model.state_dict(), os.path.join(save_path, f"model_{epoch}.pt")) 

if __name__ == "__main__":
    if mode == "train":
        exp_params = {
                     "train_size": train_dataset.__len__(),
                     "test_size": test_dataset.__len__(),
                     "use_augmentation": args.use_augment,
                     "augmentation_types": augment_type_str,
                     "aug_size": args.aug_size,
                     "gpu_num": args.gpu_num,
                     "num_epochs": args.num_epochs,
                     "train_bs": args.train_bs,
                     "test_bs": args.test_bs,
                     "val_bs": args.val_bs,
                     "resize_px": args.resize_px,
                     "val_check": args.val_check,
                     "lr": args.lr,
                     "beta1": args.beta1
        }

        print(f"\n*** [STARTING EXPERIMENT] Visualizations and best models saved at: {save_path} ***\n")
        print("[EXPERIMENT PARAMETERS]\n", json.dumps(exp_params, indent = 4))
        with open(os.path.join(save_path, 'exp_params.json'), 'w') as fp:
            json.dump(exp_params, fp, indent = 4)

        train()
        
    elif mode == "test":
        model = MTL_UNet_Main(in_channels = 1, out_dict = {'class': 15, 'image': 1})
        weights = torch.load(model_path)
        model.load_state_dict(weights)
        model = model.to(device)    
        
        iou, dsc, custom_acc, pixel_acc, binary_acc, cur_loss, accs, f1s = test(model)
        with open(f"{save_path}/result.txt", "w") as file:
            text = "[TEST-FINAL] IoU: %2.3f, DSC: %2.3f, aACC: %.3f, pACC: %3.3f, bACC: %3.3f, acc: %3.3f, f1: %3.3f" % (iou, dsc, custom_acc, pixel_acc, binary_acc, accs, f1s)
            file.write(text)
        print("[TEST-FINAL] IoU: %2.3f, DSC: %2.3f, aACC: %.3f, pACC: %3.3f, bACC: %3.3f, acc: %3.3f, f1: %3.3f" % (iou, dsc, custom_acc, pixel_acc, binary_acc, accs, f1s))

    elif mode == "val":
        model = MTL_UNet_Main(in_channels = 1, out_dict = {'class': 15, 'image': 1})
        for i in range(num_epochs):
            model_path = save_path + f"model_{i}.pt"
            weights = torch.load(model_path)
            model.load_state_dict(weights)
            model = model.to(device)    
            
            iou, dsc, custom_acc, pixel_acc, binary_acc, cur_loss, accs, f1s = test(model, tset="Valid")
            with open(f"{save_path}/result_valid.txt", "a") as file:
                text = "[TEST-FINAL] IoU: %2.3f, DSC: %2.3f, pACC: %3.3f, aACC: %.3f, bACC: %3.3f, acc: %.3f, f1: %3.3f" % (iou, dsc, pixel_acc, custom_acc, binary_acc, accs, f1s)
                file.write(text)
            print("[TEST-FINAL] IoU: %2.3f, DSC: %2.3f, aACC: %.3f, pACC: %3.3f, bACC: %3.3f, acc: %.3f, f1: %3.3f" % (iou, dsc, custom_acc, pixel_acc, binary_acc, accs, f1s))