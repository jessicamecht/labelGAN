import torch 
import os 
import torch.nn as nn 
import numpy as np 
from utils.utils import process_image, colorize_mask, oht_to_scalar, latent_to_image
import sys
sys.path.append('..')
import imageio
torch.manual_seed(0)
from torchvision.utils import save_image
import random
import scipy.misc
from PIL import Image
import pickle
from torch.distributions import Categorical
import scipy.stats
from tqdm import tqdm
from train_dataset import *
from label_model import *

    

def generate_data(args, checkpoint_path_segm, checkpoint_path_label, num_sample, start_step=0, vis=True, device="cuda"):
    print("IN GENERATE DATA")
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
        result_path = os.path.join(checkpoint_path_segm, 'vis_test_imgs_vinbig_%d'%num_sample)
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
        print(args['number_class'], args['dim'][-1])
        classifier = segm_classifier(args['number_class']
                                      , args['dim'][-1])

        #feat_size = args['classification_map_size']*args['classification_map_size']*args['classification_channels']
        #label_classifier_instance = label_classifier(args['number_class_classification'], feat_size).to(device)
        label_classifier_instance = latent_classifier(args['number_class_classification'])

        checkpoint = torch.load(os.path.join(checkpoint_path_segm, 'model_' + '.pth'))
        sd = checkpoint['model_state_dict']
        classifier.load_state_dict(sd)
        classifier = classifier
        #classifier =  nn.DataParallel(classifier)
    
        checkpoint = torch.load(os.path.join(checkpoint_path_label, 'model_label_classif_number_.pth'))
        label_classifier_instance.load_state_dict(checkpoint['model_state_dict'])
        #label_classifier_instance =  nn.DataParallel(label_classifier_instance).to(device)
        label_classifier_instance = label_classifier_instance.to(device)


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
        style_latents_flag = args['annotation_data_from_w']

        if style_latents_flag: 
            p = args['annotation_image_latent_path_classification_generate']
            files = os.listdir(p)
            random.shuffle(files)
            #mask = ["tophat" in elem for elem in files]
            files = np.array(files)#[mask]
            num_sample = len(files)
        else:
            print("num_sample", num_sample)
        
        for i in tqdm(range(num_sample)):
            if i % 100 == 0:
                print("Generate", i, "Out of:", num_sample)

            curr_result = {}
            if style_latents_flag:
                print(p + files[i])
                latent = np.load(p + files[i])#.to(device)
            else: 
                latent = np.random.randn(1, 512)

            curr_result['latent'] = latent
            latent = torch.from_numpy(latent).type(torch.FloatTensor).to(device)
            latent_cache.append(latent)
            style_latents = latent if style_latents_flag else g_all.truncation(g_all.g_mapping(latent))
            '''img, affine_layers, style_latents, _ = latent_to_image(g_all, upsamplers, latent, dim=args['dim'][1], process_out=False,
                                                     use_style_latents=style_latents_flag, return_upsampled_layers=True, device=device)
            print('style_latents', style_latents.shape)
            if args['dim'][0] != args['dim'][1]:
                img = img[:, 64:448][0]
            else:
                img = img[0]
            img = img[0]
            image_cache.append(img)
            if args['dim'][0] != args['dim'][1]:
                affine_layers = affine_layers[:, :, 64:448]'''
            img, affine_layers = latent_to_image(g_all, upsamplers, latent, dim=args['dim'][1],
                                                         return_upsampled_layers=True, 
                                                         use_style_latents=style_latents_flag)

            #if args['dim'][0] != args['dim'][1]:
            #    img = img[:, 64:448][0]
            #else:
            
            affine_layers = affine_layers[0]
            affine_layers = affine_layers.reshape(args['dim'][-1], -1).transpose(1, 0)

            all_seg = []
            all_entropy = []
            mean_seg = None

            seg_mode_ensemble = []
            for MODEL_NUMBER in range(args['model_num']):
                classifier = classifier_list[MODEL_NUMBER]
                img_seg = classifier(affine_layers)

                classifier_label = classifier_list_label[MODEL_NUMBER]
                print(latent.shape)
                label = classifier_label(style_latents)
                label = softmax_f(label).argmax(-1).squeeze()

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
                add = files[i] if style_latents_flag else "_KDE_"
                imageio.imwrite(os.path.join(result_path, f"{label}_{add}" + str(i) + '_mask.jpg'),
                                  color_mask.astype(np.uint8))

                img = img[0]
                img = Image.fromarray(img)

                image_name =  os.path.join(result_path, f"{label}_{add}" + str(i) + '_image.jpg')
                img.save(image_name)

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