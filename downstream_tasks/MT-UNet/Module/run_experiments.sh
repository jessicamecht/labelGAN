
python3 MT_unet-rahul.py -gpu_num 1 -use_augment -train_bs 35 -aug_size 25 -resize_px 256 -aug_types Embedded KDE -mode train 
echo "Finished KDE EMB 25"
python3 MT_unet-rahul.py -gpu_num 1 -use_augment -train_bs 35 -aug_size 100 -resize_px 256 -aug_types Embedded KDE -mode train 
echo "Finished KDE EMB 100"
python3 MT_unet-rahul.py -gpu_num 1 -use_augment -train_bs 35 -aug_size 200 -resize_px 256 -aug_types Embedded KDE -mode train 
echo "Finished KDE EMB 200"

python3 MT_unet-rahul.py -gpu_num 1 -use_augment -train_bs 35 -aug_size 50 -resize_px 256 -aug_types KDE -mode train 
echo "Finished KDE 50"
python3 MT_unet-rahul.py -gpu_num 1 -use_augment -train_bs 35 -aug_size 200 -resize_px 256 -aug_types KDE -mode train 
echo "Finished KDE 200"
python3 MT_unet-rahul.py -gpu_num 1 -use_augment -train_bs 35 -aug_size 400 -resize_px 256 -aug_types KDE -mode train 
echo "Finished KDE 400"

python3 MT_unet-rahul.py -gpu_num 1 -use_augment -train_bs 35 -aug_size 50 -resize_px 256 -aug_types Embedded -mode train 
echo "Finished  EMB 50"

python3 MT_unet-rahul.py -gpu_num 1 -use_augment -train_bs 35 -aug_size 200 -resize_px 256 -aug_types Embedded -mode train 
echo "Finished EMB 200"


python3 MT_unet-rahul.py -gpu_num 1 -use_augment -train_bs 35 -aug_size 400 -resize_px 256 -aug_types Embedded -mode train 
echo "Finished  EMB 400"

python3 MT_unet-rahul.py -gpu_num 1  -train_bs 35 -resize_px 256 -aug_types baseline -mode train 
echo "Finished baseline"

python3 MT_unet-rahul.py -gpu_num 1 -use_augment -train_bs 35 -aug_size 50 -resize_px 256 -aug_types baseline -mode train 
echo "Finished basel 50"
python3 MT_unet-rahul.py -gpu_num 1 -use_augment -train_bs 35 -aug_size 400 -resize_px 256 -aug_types baseline -mode train 
echo "Finished basel 400" 
python3 MT_unet-rahul.py -gpu_num 1 -use_augment -train_bs 35 -aug_size 200 -resize_px 256 -aug_types baseline -mode train 
echo "Finished basel 200"

