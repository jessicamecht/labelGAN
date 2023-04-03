import torch 
import os 
import torch.nn as nn 
import numpy as np 
from utils.utils import colorize_mask, latent_to_image, oht_to_scalar
import sys
sys.path.append('..')
import imageio
torch.manual_seed(0)
import scipy.misc
from PIL import Image
import pickle
from torch.distributions import Categorical
import scipy.stats
from train_dataset import *
from label_model import *

def generate_data(args, checkpoint_path_segm, checkpoint_path_label, num_sample, style_latents=False, start_step=0, vis=True, device="cuda"):
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
        #label_classifier_instance = label_classifier(args['number_class_classification'], feat_size).to(device)
        label_classifier_instance = latent_classifier(args['number_class_classification']).to(device)
        label_classifier_instance =  nn.DataParallel(label_classifier_instance).to(device)

        checkpoint = torch.load(os.path.join(checkpoint_path_segm, 'model_' + str(MODEL_NUMBER) + '.pth'))
        classifier.load_state_dict(checkpoint['model_state_dict'])
    
        checkpoint = torch.load(os.path.join(checkpoint_path_label, 'model_label_classif_20000_number_0.pth'))
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

        if style_latents: 
            p = args['annotation_image_path_classification']
            files = os.listdir(p)
            mask = ["tophat" in elem for elem in files]
            files = np.array(files)[mask]
            num_sample = len(files)
        
        for i in range(num_sample):
            if i % 100 == 0:
                print("Generate", i, "Out of:", num_sample)

            curr_result = {}
            if style_latents:
                latent = np.load(files[i])
            else: 
                latent = np.random.randn(1, 512)

            curr_result['latent'] = latent


            latent = torch.from_numpy(latent).type(torch.FloatTensor).to(device)
            latent_cache.append(latent)
            
            img, affine_layers, style_latents = latent_to_image(g_all, upsamplers, latent, dim=args['dim'][1],
                                                     style_latents=style_latents, return_upsampled_layers=True, device=device)
            
            if args['dim'][0] != args['dim'][1]:
                img = img[0]#img[:, 64:448][0]
            else:
                img = img[0]

            image_cache.append(img)
            #if args['dim'][0] != args['dim'][1]:
            #    affine_layers = affine_layers[:, :, 64:448]
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

                label = classifier_label(latent)
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
                
                imageio.imwrite(os.path.join(result_path, f"vis_{label}_" + str(i) + '_mask.jpg'),
                                  color_mask.astype(np.uint8))
                
                imageio.imwrite(os.path.join(result_path, f"vis_{label}_" + str(i) + '_image.jpg'),
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