import torch 
import math 
from PIL import Image
from torchvision.utils import save_image
from torchvision import transforms
import matplotlib.pyplot as plt 
import torch.optim as optim
from model import * 
import numpy as np
import torch.nn as nn

def PSNR(mse, flag = 0):
  #flag = 0 if a single image is used and 1 if loss for a batch of images is to be calculated
  if flag == 0:
    psnr = 10 * math.log10(1 / mse.item())
  return psnr

def read_image(img_path, device="cpu"):
  with open(img_path,"rb") as f: 
    image=Image.open(f)
    image=image.convert("RGB")
  transform = transforms.Compose([transforms.ToTensor()])
  image = transform(image)
  image = image.unsqueeze(0)
  image = image.to(device)
  return image

def save_random_generations(g_all, device):
  for i in range(20):
    z = torch.randn(1,512,device = device)
    img = g_all(z, depth=8, alpha=0)
    img = (img +1.0)/2.0
    save_image(img.clamp(0,1),"save_image/random_SG1-{}.png".format(i+1))

def embedding_function(image, perceptual, g_synthesis, g_mapping, avg_latent, device, image_id, save_path):
  upsample = torch.nn.Upsample(scale_factor = 256/1024, mode = 'bilinear')
  tr = transforms.Resize((1024, 1024))
  image = tr(image)
  img_p = image.clone()
  img_p = upsample(img_p)
  
  #MSE loss object
  MSE_loss = nn.MSELoss(reduction="mean")
  #since the synthesis network expects 18 w vectors of size 1x512 thus we take latent vector of the same size
  latents = torch.zeros((1,18,512), requires_grad = True, device = device)

  #latents_z = torch.zeros((1,512), requires_grad = True, device = device)
  #Optimizer to change latent code in each backward step
  optimizer = optim.Adam({latents},lr=0.01,betas=(0.9,0.999),eps=1e-8)



  #Loop to optimise latent vector to match the generated image to input image
  loss_ = []
  loss_psnr = []
  for e in range(1500):
    #res = g_mapping(latents_z)
    #x = res
    #interp = torch.lerp(avg_latent, x, 0.7).to(device)
    #do_trunc = (torch.arange(x.size(1)) < 9).view(1, -1, 1).to(device)
    #latents = torch.where(do_trunc, interp, x)
    optimizer.zero_grad()
    syn_img, _ = g_synthesis(latents)#, depth=8)
    #syn_img = (syn_img+1.0)/2.0
    mse, per_loss = loss_function(syn_img, image, img_p, MSE_loss, upsample, perceptual)
    
    psnr = PSNR(mse, flag = 0)
    loss = per_loss +mse
    loss.backward()
    
    optimizer.step()
    loss_np=loss.detach().cpu().numpy()
    loss_p=per_loss.detach().cpu().numpy()
    loss_m=mse.detach().cpu().numpy()
    loss_psnr.append(psnr)
    loss_.append(loss_np)
    #if (e+1)%500==0 :
    #  print("iter{}: loss -- {},  mse_loss --{},  percep_loss --{}, psnr --{}".format(e+1,loss_np,loss_m,loss_p,psnr))
  #save_image(torch.tensor(syn_img),f"./images/generated/{image_id}_.png".format(e+1))
  save_image(syn_img.clamp(0,1),f"{save_path}/generated/{image_id}.png".format(e+1))
  #np.save("loss_list.npy",loss_)
  np.save(f"{save_path}/generated/latent_{image_id}.npy".format(),latents.detach().cpu().numpy())
  #np.save(f"{save_path}/generated/512_latent_{image_id}.npy".format(),latents.detach().cpu().numpy())

  #plt.plot(loss_, label = 'Loss = MSELoss + Perceptual')
  #plt.plot(loss_psnr, label = 'PSNR')
  #plt.legend()
  return latents, syn_img

def embedding_Hierarchical(image, perceptual, g_synthesis, device):
  upsample = torch.nn.Upsample(scale_factor = 256/1024, mode = 'bilinear')
  tr = transforms.Resize((1024, 1024))
  image = tr(image)
  img_p = image.clone()
  img_p = upsample(img_p)
  
  #MSE loss object
  MSE_loss = nn.MSELoss(reduction="mean")
  #since the synthesis network expects 18 w vectors of size 1x512 thus we take latent vector of the same size
  latent_w = torch.zeros((1,512), requires_grad = True, device = device)
  
  #Optimizer to change latent code in each backward step
  optimizer = optim.Adam({latent_w},lr=0.01,betas=(0.9,0.999),eps=1e-8)


  #Loop to optimise latent vector to match the generated image to input image
  loss_ = []
  loss_psnr = []
  for e in range(1000):
    optimizer.zero_grad()
    latent_w1 = latent_w.unsqueeze(1).expand(-1, 18, -1)
    syn_img = g_synthesis(latent_w1)#, depth=8)
    syn_img = (syn_img+1.0)/2.0
    mse, per_loss = loss_function(syn_img, image, img_p, MSE_loss, upsample, perceptual)
    psnr = PSNR(mse, flag = 0)
    loss = per_loss +mse
    loss.backward()
    optimizer.step()
    loss_np=loss.detach().cpu().numpy()
    loss_p=per_loss.detach().cpu().numpy()
    loss_m=mse.detach().cpu().numpy()
    loss_psnr.append(psnr)
    loss_.append(loss_np)
    if (e+1)%500==0 :
      print("iter{}: loss -- {},  mse_loss --{},  percep_loss --{}, psnr --{}".format(e+1,loss_np,loss_m,loss_p,psnr))
      save_image(syn_img.clamp(0,1),"Hier_pass_morphP1-{}.png".format(e+1))
      #np.save("loss_list.npy",loss_)
      #np.save("latent_W.npy".format(),latents.detach().cpu().numpy())

  
  latent_w1 = latent_w.unsqueeze(1).expand(-1, 18, -1)
  latent_w1 = torch.tensor(latent_w1, requires_grad=True)
  optimizer = optim.Adam({latent_w1},lr=0.01,betas=(0.9,0.999),eps=1e-8)
  for e in range(1000):  
    optimizer.zero_grad()
    syn_img = g_synthesis(latent_w1)#, depth=8)
    syn_img = (syn_img+1.0)/2.0
    mse, per_loss = loss_function(syn_img, image, img_p, MSE_loss, upsample, perceptual)
    psnr = PSNR(mse, flag = 0)
    loss = per_loss +mse
    loss.backward()
    optimizer.step()
    loss_np=loss.detach().cpu().numpy()
    loss_p=per_loss.detach().cpu().numpy()
    loss_m=mse.detach().cpu().numpy()
    loss_psnr.append(psnr)
    loss_.append(loss_np)
    if (e+1)%500==0 :
      print("iter{}: loss -- {},  mse_loss --{},  percep_loss --{}, psnr --{}".format(e+1,loss_np,loss_m,loss_p,psnr))
      save_image(syn_img.clamp(0,1),"Hier_pass_morphP2-{}.png".format(e+1))


  #plt.plot(loss_, label = 'Loss = MSELoss + Perceptual')
  #plt.plot(loss_psnr, label = 'PSNR')
  #plt.legend()
  return latent_w1