import torch 
import os 
import torch.nn as nn 
import numpy as np 
from utils.utils import colorize_mask, latent_to_image, oht_to_scalar
import os
import sys
sys.path.append('..')
import imageio
import torch
import torch.nn as nn
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
device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
from train_dataset import *
from label_model import *
import cv2

class trainData(Dataset):

    def __init__(self, X_data, y_data ):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return torch.tensor(self.X_data[index]).type(torch.FloatTensor), torch.tensor(self.y_data[index]).type(torch.FloatTensor)

    def __len__(self):
        return len(self.X_data)
    
class labelData(Dataset):
    def __init__(self, X_data, labels ):
        self.X_data = X_data
        self.labels = labels
        print('X_data', X_data.shape, labels.shape)

    def __getitem__(self, index):
        return torch.tensor(self.X_data[index]).type(torch.FloatTensor), torch.tensor(self.labels[index]).type(torch.FloatTensor)

    def __len__(self):
        return len(self.X_data)
    

def generate_data(args, checkpoint_path, num_sample, start_step=0, vis=True):
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
        result_path = os.path.join(checkpoint_path, 'samples' )
    else:
        result_path = os.path.join(checkpoint_path, 'vis_%d'%num_sample)
    if os.path.exists(result_path):
        pass
    else:
        os.system('mkdir -p %s' % (result_path))
        print('Experiment folder created at: %s' % (result_path))


    g_all, avg_latent, upsamplers = prepare_stylegan(args)

    classifier_list = []
    for MODEL_NUMBER in range(args['model_num']):
        print('MODEL_NUMBER', MODEL_NUMBER)

        classifier = label_classifier(args['number_class']
                                      , args['dim'][-1])
        classifier =  nn.DataParallel(classifier).to(device)

        checkpoint = torch.load(os.path.join(checkpoint_path, 'model_' + str(MODEL_NUMBER) + '.pth'))

        classifier.load_state_dict(checkpoint['model_state_dict'])


        classifier.eval()
        classifier_list.append(classifier)
    
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
            
            img, affine_layers = latent_to_image(g_all, upsamplers, latent, dim=args['dim'][1],
                                                     return_upsampled_layers=True)
            
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

def prepare_data(args, palette):
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

    g_all, avg_latent, upsamplers = prepare_stylegan(args)
    latent_all = np.load(args['annotation_image_latent_path'])
    latent_classification = np.load(args['annotation_image_latent_path'])
    
    latent_all = torch.from_numpy(latent_all)
    latent_classification = torch.from_numpy(latent_classification)

    # load annotated mask
    mask_list = []
    im_list = []
    label_list = []
    latent_all = latent_all[:args['max_training']]
    latent_classification = latent_classification[:args['max_training']]
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
        label_list.append(few_shot_classes["None"])

    '''image_names_classification = os.listdir(args['annotation_image_path_classification'])
    for i, imagename in enumerate(image_names_classification):
        
        if i >= args['max_training']:
            break
        
        im_name = os.path.join(args['annotation_image_path_classification'], imagename)
        img = Image.open(im_name)
        img = img.resize((args['dim'][1], args['dim'][0]))
        mask_list.append(np.zeros((np.array(img).shape)))  
        im_list.append(np.array(img))
        label_list.append(few_shot_classes[imagename.split("_")[0]])
    # delete small annotation error
    for i in range(len(mask_list)):  # clean up artifacts in the annotation, must do
        if mask_list[i] == None: continue 
        for target in range(1, 50):
            if (mask_list[i] == target).sum() < 30:
                mask_list[i][mask_list[i] == target] = 0'''


    all_mask = np.stack(mask_list)

    #latent_all = np.concatenate((latent_all, latent_classification), axis=0)
    # 3. Generate ALL training data for training pixel classifier
    #all_feature_maps_train = np.zeros((args['dim'][0] * args['dim'][1] * len(latent_all), args['dim'][2]), dtype=np.float16)
    all_feature_maps_train_list = []
    vis = []
    all_mask_train_list = []
    for i in range(len(latent_all)):

        gc.collect()

        latent_input = latent_all[i].float()

        img, feature_maps = latent_to_image(g_all, upsamplers, latent_input.unsqueeze(0), dim=args['dim'][1],
                                            return_upsampled_layers=True, use_style_latents=args['annotation_data_from_w'])

        mask = all_mask[i:i + 1]
        feature_maps = feature_maps.permute(0, 2, 3, 1)
        print(feature_maps.shape)
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
    all_feature_maps_train = torch.concat(all_mask_train_list, axis=0)
    vis = np.concatenate(vis, 1)
    import imageio
    imageio.imwrite(os.path.join(args['exp_dir'], "train_data.jpg"),
                      vis)

    return all_feature_maps_train_list, all_mask_train_list, num_data, np.array(label_list)

