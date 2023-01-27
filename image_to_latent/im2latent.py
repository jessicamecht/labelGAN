import torch
from torchvision.models import resnet50
from PIL import Image
import numpy as np

class ImageToLatent(torch.nn.Module):
    def __init__(self, image_size=256):
        super().__init__()
        
        self.image_size = image_size
        self.activation = torch.nn.ELU()
        
        self.resnet = list(resnet50(pretrained=True).children())[:-2]
        self.resnet = torch.nn.Sequential(*self.resnet)
        self.conv2d = torch.nn.Conv2d(2048, 256, kernel_size=1)
        self.flatten = torch.nn.Flatten()
        self.dense1 = torch.nn.Linear(16384, 256)
        self.dense2 = torch.nn.Linear(256, (18 * 512))

    def forward(self, image):
        x = self.resnet(image)
        x = self.conv2d(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = x.view((-1, 18, 512))

        return x

class ImageLatentDataset(torch.utils.data.Dataset):
    def __init__(self, filenames, dlatents, image_size=256, transforms = None):
        self.filenames = filenames
        self.dlatents = dlatents
        self.image_size = image_size
        self.transforms = transforms

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        dlatent = self.dlatents[index]

        image = self.load_image(filename)
        image = Image.fromarray(np.uint8(image))

        if self.transforms:
            image = self.transforms(image)

        return image, dlatent

    def load_image(self, filename):
        image = np.asarray(Image.open(filename))

        return image

class LatentLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = L1Loss()
        self.log_cosh_loss = LogCoshLoss()
        self.l2_loss = torch.nn.MSELoss()
    
    def forward(self, real_features, generated_features, average_dlatents = None, dlatents = None):
        # Take a look at:
            # https://github.com/pbaylies/stylegan-encoder/blob/master/encoder/perceptual_model.py
            # For additional losses and practical scaling factors.
        loss = 0      
        loss += 1 * self.l2_loss(real_features, generated_features)
        if average_dlatents is not None and dlatents is not None:
            loss += 1 * 512 * self.l1_loss(average_dlatents, dlatents)

        return loss

class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, true, pred):
        loss = true - pred
        return torch.mean(torch.log(torch.cosh(loss + 1e-12)))
    
class L1Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, true, pred):
        return torch.mean(torch.abs(true - pred))
import torch.nn.functional as F
from torchvision.models import vgg16
import torch

class PostSynthesisProcessing(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.min_value = -1
        self.max_value = 1

    def forward(self, synthesized_image):
        synthesized_image = (synthesized_image - self.min_value) * torch.tensor(255).float() / (self.max_value - self.min_value)
        synthesized_image = torch.clamp(synthesized_image + 0.5, min=0, max=255)

        return synthesized_image

class VGGProcessing(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.image_size = 256
        self.mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(-1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(-1, 1, 1)

    def forward(self, image):
        image = image / torch.tensor(255).float()
        image = F.adaptive_avg_pool2d(image, self.image_size)

        image = (image - self.mean) / self.std

        return image


class LatentOptimizer(torch.nn.Module):
    def __init__(self, synthesizer, layer=12):
        super().__init__()

        self.synthesizer = synthesizer.cuda().eval()
        self.post_synthesis_processing = PostSynthesisProcessing()
        self.vgg_processing = VGGProcessing()
        self.vgg16 = vgg16(pretrained=True).features[:layer].cuda().eval()


    def forward(self, dlatents):
        generated_image = self.synthesizer(dlatents)
        generated_image = self.post_synthesis_processing(generated_image)
        generated_image = self.vgg_processing(generated_image)
        features = self.vgg16(generated_image)

        return features