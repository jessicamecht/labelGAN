import torch 
import os 
import torch.nn as nn 
import numpy as np 
from utils.utils import colorize_mask, latent_to_image, oht_to_scalar
import os
import sys
sys.path.append('..')
import imageio
from tqdm import tqdm
import torch
import torch.nn as nn
import imageio
torch.manual_seed(0)
import scipy.misc
import numpy as np
import os
from PIL import Image
import gc
import pickle
from torch.distributions import Categorical
import scipy.stats
from torch.utils.data import Dataset
from train_dataset import *
from label_model import *
import torchvision.transforms as transforms
import cv2

few_shot_classes = {"None": 0,
    'NoduleMass': 1,
    'Infiltration': 2,
    'LungOpacity': 3,
    'Consolidation': 4,
    'Pleuralthickening': 5,
    'ILD': 6,
    'Cardiomegaly': 7,
    'Pulmonaryfibrosis': 8,
    'Aorticenlargement': 9,
    'Otherlesion': 10,
    'Pleuraleffusion': 11,
    'Calcification': 12,
    'Atelectasis': 13,
    'Pneumothorax': 14,
    'Nofinding': 15}


class trainData(Dataset):

    def __init__(self, X_data, y_data, args):
        self.X_data = X_data
        self.y_data = y_data
        self.args = args

    def __getitem__(self, index):
        x = torch.tensor(self.X_data[index])
        x = x.reshape(-1, self.args['dim'][2])
        return x.type(torch.FloatTensor).squeeze(), torch.tensor(self.y_data[index]).type(torch.FloatTensor)

    def __len__(self):
        return len(self.X_data)
    
class labelData(Dataset):
    def __init__(self, imagenames, labels, args):
        self.imagenames = imagenames
        self.args = args
        self.labels = labels

    def __getitem__(self, index):
        x = self.imagenames[index]
        #im_name = os.path.join(self.args['annotation_image_path_classification'], x)
        #x = torch.tensor(np.load(im_name)).type(torch.FloatTensor)
        y = self.labels[index]#torch.tensor([few_shot_classes[imagename.split("_")[0]]])
        return x,y

    def __len__(self):
        return len(self.imagenames)
    

def generate_data(args, checkpoint_path_segm, checkpoint_path_label, num_sample, start_step=0, vis=True, device="cuda"):
    if args['category'] == 'car':
        from utils.data_util import car_20_palette as palette
    elif args['category'] == 'face':
        from utils.data_util import face_palette as palette
    elif args['category'] == 'bedroom':
        from utils.data_util import bedroom_palette as palette
    elif args['category'] == 'cat':
        from utils.data_util import cat_palette as palette
    elif args['category'] == 'xray':
        from utils.data_util import xray_palette as palette
    else:
        assert False
    if not vis:
        result_path = os.path.join(checkpoint_path_segm, 'samples' )
    else:
        result_path = os.path.join(checkpoint_path_segm, 'vis_%d'%num_sample)
    if os.path.exists(result_path):
        pass
    else:
        os.system('mkdir -p %s' % (result_path))
        print('Experiment folder created at: %s' % (result_path))


    g_all, avg_latent, upsamplers = prepare_stylegan(args, device)

    classifier_list = []
    classifier_list_label = []
    for MODEL_NUMBER in range(args['model_num']):
        print('MODEL_NUMBER', MODEL_NUMBER)

        classifier = segm_classifier(args['number_class']
                                      , args['dim'][-1])
        classifier =  nn.DataParallel(classifier).to(device)

        feat_size = args['classification_map_size']*args['classification_map_size']*args['classification_channels']
        label_classifier_instance = label_classifier(args['number_class_classification'], feat_size).to(device)
        label_classifier_instance =  nn.DataParallel(label_classifier_instance).to(device)

        checkpoint = torch.load(os.path.join(checkpoint_path_segm, 'model_' + str(MODEL_NUMBER) + '.pth'))
        classifier.load_state_dict(checkpoint['model_state_dict'])
    
        checkpoint = torch.load(os.path.join(checkpoint_path_label, 'model_label_classif_' + str(MODEL_NUMBER) + '.pth'))
        label_classifier_instance.load_state_dict(checkpoint['model_state_dict'])


        classifier.eval()
        label_classifier_instance.eval()
        classifier_list.append(classifier)
        classifier_list_label.append(label_classifier_instance)
    
    softmax_f = nn.Softmax(dim=1)
    with torch.no_grad():
        latent_cache = []
        image_cache = []
        seg_cache = []
        entropy_calculate = []
        results = []
        np.random.seed(start_step)
        count_step = start_step



        print( "num_sample: ", num_sample)
        
        for i in range(num_sample):
            if i % 100 == 0:
                print("Generate", i, "Out of:", num_sample)

            curr_result = {}

            latent = np.random.randn(1, 512)

            curr_result['latent'] = latent


            latent = torch.from_numpy(latent).type(torch.FloatTensor).to(device)
            latent_cache.append(latent)
            
            img, affine_layers, style_latents = latent_to_image(g_all, upsamplers, latent, dim=args['dim'][1],
                                                     return_upsampled_layers=True, device=device)
            
            if args['dim'][0] != args['dim'][1]:
                img = img[:, 64:448][0]
            else:
                img = img[0]

            image_cache.append(img)
            if args['dim'][0] != args['dim'][1]:
                affine_layers = affine_layers[:, :, 64:448]
            affine_layers = affine_layers[0]

            affine_layers = affine_layers.reshape(args['dim'][-1], -1).transpose(1, 0)

            all_seg = []
            all_entropy = []
            mean_seg = None

            seg_mode_ensemble = []
            for MODEL_NUMBER in range(args['model_num']):
                classifier = classifier_list[MODEL_NUMBER]
                img_seg = classifier(affine_layers)

                img_seg = img_seg.squeeze()


                entropy = Categorical(logits=img_seg).entropy()
                all_entropy.append(entropy)

                all_seg.append(img_seg)
                if mean_seg is None:
                    mean_seg = softmax_f(img_seg)
                else:
                    mean_seg += softmax_f(img_seg)

                img_seg_final = oht_to_scalar(img_seg)
                img_seg_final = img_seg_final.reshape(args['dim'][0], args['dim'][1], 1)
                img_seg_final = img_seg_final.cpu().detach().numpy()

                seg_mode_ensemble.append(img_seg_final)

            mean_seg = mean_seg / len(all_seg)

            full_entropy = Categorical(mean_seg).entropy()

            js = full_entropy - torch.mean(torch.stack(all_entropy), 0)

            top_k = js.sort()[0][- int(js.shape[0] / 10):].mean()
            entropy_calculate.append(top_k)


            img_seg_final = np.concatenate(seg_mode_ensemble, axis=-1)
            img_seg_final = scipy.stats.mode(img_seg_final, 2)[0].reshape(args['dim'][0], args['dim'][1])
            del (affine_layers)
            if vis:

                color_mask = colorize_mask(img_seg_final, palette) #+ 0.3 * img
                
                imageio.imwrite(os.path.join(result_path, "vis_" + str(i) + '_mask.jpg'),
                                  color_mask.astype(np.uint8))
                
                imageio.imwrite(os.path.join(result_path, "vis_" + str(i) + '_image.jpg'),
                                  img.astype(np.uint8))

            else:
                seg_cache.append(img_seg_final)
                curr_result['uncertrainty_score'] = top_k.item()
                image_label_name = os.path.join(result_path, 'label_' + str(count_step) + '.png')
                image_name = os.path.join(result_path,  str(count_step) + '.png')

                js_name = os.path.join(result_path, str(count_step) + '.npy')
                img = Image.fromarray(img)
                img_seg = Image.fromarray(img_seg_final.astype('uint8'))
                js = js.cpu().numpy().reshape(args['dim'][0], args['dim'][1])
                img.save(image_name)
                img_seg.save(image_label_name)
                np.save(js_name, js)
                curr_result['image_name'] = image_name
                curr_result['image_label_name'] = image_label_name
                curr_result['js_name'] = js_name
                count_step += 1


                results.append(curr_result)
                if i % 1000 == 0 and i != 0:
                    with open(os.path.join(result_path, str(i) + "_" + str(start_step) + '.pickle'), 'wb') as f:
                        pickle.dump(results, f)

        with open(os.path.join(result_path, str(num_sample) + "_" + str(start_step) + '.pickle'), 'wb') as f:
            pickle.dump(results, f)

def prepare_data(args, palette, device):

    g_all, avg_latent, upsamplers = prepare_stylegan(args, device)
    latent_all = np.load(args['annotation_image_latent_path'])
    
    latent_all = torch.from_numpy(latent_all)
    

    # load annotated mask
    mask_list = []
    im_list = []
    latent_all = latent_all[:args['max_training']]
    
    num_data = len(latent_all)

    for i in range(len(latent_all)):

        if i == 3: continue
        if i >= args['max_training']:
            break
        name = 'image_%0d.png' % i
        
        im_frame = Image.open(os.path.join( args['annotation_mask_path'] , name)).convert('L')
        mask = np.array(im_frame)
        mask = mask.squeeze()
        mask =  cv2.resize(np.float32(mask), dsize=(args['dim'][1], args['dim'][0]), interpolation=cv2.INTER_NEAREST)
        mask_list.append(mask)
        im_name = os.path.join( args['annotation_mask_path'], 'image_%d.jpg' % i)
        img = Image.open(im_name)
        img = img.resize((args['dim'][1], args['dim'][0]))

        im_list.append(np.array(img))
    image_names_classification = os.listdir(args['annotation_image_path_classification'])
    affine_layers_list = []
    labels = []
    for x in image_names_classification:
        im_name = os.path.join(args['annotation_image_path_classification'], x)
        label = torch.tensor([few_shot_classes[x.split("_")[0]]])
        latent_input = torch.tensor(np.load(im_name)).type(torch.FloatTensor).to(device).squeeze()
        img, feature_maps, style_latents, affine_layers = latent_to_image(g_all, upsamplers, latent_input.unsqueeze(0), dim=args['dim'][1],
                                            return_upsampled_layers=True, use_style_latents=True, device=device)
        affine_layers_list.extend([elem.cpu().detach()for elem in affine_layers])
        labels.extend([label] * len(affine_layers))
    #image_names_classification = image_names_classification[:args['max_training']]
    # delete small annotation error
    '''for i in range(len(mask_list)):  # clean up artifacts in the annotation, must do
        if mask_list[i] == None: continue 
        for target in range(1, 50):
            if (mask_list[i] == target).sum() < 30:
                mask_list[i][mask_list[i] == target] = 0'''

    all_mask = np.stack(mask_list)
    all_feature_maps_train_list = []
    vis = []
    all_mask_train_list = []
    for i in tqdm(range(len(latent_all))):
        gc.collect()
        latent_input = latent_all[i].float()

        img, feature_maps, style_latents, affine_layers = latent_to_image(g_all, upsamplers, latent_input.unsqueeze(0), dim=args['dim'][1],
                                            return_upsampled_layers=True, use_style_latents=args['annotation_data_from_w'], device=device)

        #print('test', feature_maps.shape)
        mask = all_mask[i:i + 1]
        feature_maps = feature_maps.permute(0, 2, 3, 1)
        feature_maps = feature_maps.reshape(-1, args['dim'][2])
        new_mask =  np.squeeze(mask)
        mask = mask.reshape(-1)
        #all_feature_maps_train[start:end] = feature_maps.cpu().detach().numpy().astype(np.float16)
        if len(mask) == 0: continue
        all_feature_maps_train_list.append(feature_maps.cpu().detach())
        

        #all_mask_train[start:end] = (mask == 143).astype(np.float16)
        all_mask_train_list.append(torch.tensor((mask == 143).astype(np.float16)))
        img_show =  cv2.resize(np.squeeze(img[0]), dsize=(args['dim'][1], args['dim'][1]), interpolation=cv2.INTER_NEAREST)
        curr_vis = np.concatenate( [im_list[i], img_show, colorize_mask(new_mask, palette)], 0)
        vis.append( curr_vis )
    all_feature_maps_train = torch.concat(all_feature_maps_train_list, axis=0)
    vis = np.concatenate(vis, 1)
    all_mask_train_list = torch.concat(all_mask_train_list, axis=0)
    imageio.imwrite(os.path.join(args['exp_dir'], "train_data.jpg"),
                      vis)
    return all_feature_maps_train, all_mask_train_list, num_data, image_names_classification, affine_layers_list, labels

