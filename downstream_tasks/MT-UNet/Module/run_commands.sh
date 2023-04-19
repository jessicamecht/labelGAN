#!/bin/bash 

# with augmentation = 0, no augmentation all res
# echo " ***** python MT_unet-rahul.py -gpu_num 0 -train_bs 32 -resize_px 256 -val_check 5 -num_epochs 50 ***** "
# python MT_unet-rahul.py -gpu_num 0 -train_bs 32 -resize_px 256 -val_check 5 -num_epochs 50 -mode train
# echo " ***** python MT_unet-rahul.py -gpu_num 0 -train_bs 8 -resize_px 512 -val_check 5 -num_epochs 50 ***** "
# python MT_unet-rahul.py -gpu_num 0 -train_bs 8 -resize_px 512 -val_check 5 -num_epochs 50 -mode train
# echo " ***** python MT_unet-rahul.py -gpu_num 0 -train_bs 2 -resize_px 1024 -val_check 5 -num_epochs 50 ***** "
# python MT_unet-rahul.py -gpu_num 0 -train_bs 1 -resize_px 1024 -val_check 5 -num_epochs 50 -mode train


# 256 aug
# echo " 1 ***** python MT_unet-rahul.py -gpu_num 0 -train_bs 16 -resize_px 256 -val_check 5 -num_epochs 50 -use_augment -aug_types baseline -aug_size 50 ***** "
# python MT_unet-rahul.py -gpu_num 0 -train_bs 16 -resize_px 256 -val_check 5 -num_epochs 50 -use_augment -aug_types baseline -aug_size 50 -mode train

# echo " 2 ***** python MT_unet-rahul.py -gpu_num 0 -train_bs 16 -resize_px 256 -val_check 5 -num_epochs 50 -use_augment -aug_types baseline -aug_size 200 ***** "
# python MT_unet-rahul.py -gpu_num 0 -train_bs 16 -resize_px 256 -val_check 5 -num_epochs 50 -use_augment -aug_types baseline -aug_size 200 -mode train

# echo " 3 ***** python MT_unet-rahul.py -gpu_num 0 -train_bs 16 -resize_px 256 -val_check 5 -num_epochs 50 -use_augment -aug_types baseline -aug_size 400 ***** "
# python MT_unet-rahul.py -gpu_num 0 -train_bs 16 -resize_px 256 -val_check 5 -num_epochs 50 -use_augment -aug_types baseline -aug_size 400 -mode train


# # 512 aug
# echo " 4 ***** python MT_unet-rahul.py -gpu_num 0 -train_bs 4 -resize_px 512 -val_check 5 -num_epochs 50 -use_augment -aug_types baseline -aug_size 50 ***** "
# python MT_unet-rahul.py -gpu_num 0 -train_bs 4 -resize_px 512 -val_check 5 -num_epochs 50 -use_augment -aug_types baseline -aug_size 50 -mode train

# echo " 5 ***** python MT_unet-rahul.py -gpu_num 0 -train_bs 4 -resize_px 512 -val_check 5 -num_epochs 50 -use_augment -aug_types baseline -aug_size 200 ***** "
# python MT_unet-rahul.py -gpu_num 0 -train_bs 4 -resize_px 512 -val_check 5 -num_epochs 50 -use_augment -aug_types baseline -aug_size 200 -mode train

# echo " 6 ***** python MT_unet-rahul.py -gpu_num 0 -train_bs 4 -resize_px 512 -val_check 5 -num_epochs 50 -use_augment -aug_types baseline -aug_size 400 ***** "
# python MT_unet-rahul.py -gpu_num 0 -train_bs 4 -resize_px 512 -val_check 5 -num_epochs 50 -use_augment -aug_types baseline -aug_size 400 -mode train

# 1024 aug
echo " ***** python MT_unet-rahul.py -gpu_num 0 -train_bs 1 -val_bs 1 -resize_px 1024 -val_check 5 -num_epochs 50 -mode train ***** "
python MT_unet-rahul.py -gpu_num 0 -train_bs 1 -val_bs 1 -resize_px 1024 -val_check 5 -num_epochs 50 -mode train

echo " 7 ***** python MT_unet-rahul.py -gpu_num 0 -train_bs 1 -val_bs 1 -resize_px 1024 -val_check 5 -num_epochs 50 -use_augment -aug_type baseline -aug_size 50 ***** "
python MT_unet-rahul.py -gpu_num 0 -train_bs 1 -val_bs 1 -resize_px 1024 -val_check 5 -num_epochs 50 -use_augment -aug_type baseline -aug_size 50 -mode train

echo " 8 ***** python MT_unet-rahul.py -gpu_num 0 -train_bs 1 -val_bs 1 -resize_px 1024 -val_check 5 -num_epochs 50 -use_augment -aug_type baseline -aug_size 200 ***** "
python MT_unet-rahul.py -gpu_num 0 -train_bs 1 -val_bs 1 -resize_px 1024 -val_check 5 -num_epochs 50 -use_augment -aug_type baseline -aug_size 200 -mode train

echo " 9 ***** python MT_unet-rahul.py -gpu_num 0 -train_bs 1 -val_bs 1 -resize_px 1024 -val_check 5 -num_epochs 50 -use_augment -aug_type baseline -aug_size 400 ***** "
python MT_unet-rahul.py -gpu_num 0 -train_bs 1 -val_bs 1 -resize_px 1024 -val_check 5 -num_epochs 50 -use_augment -aug_type baseline -aug_size 400 -mode train



