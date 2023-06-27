from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tf

def get_random_color_distortion(s=1.0):
    return transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    
def get_random_crop(size):
    return transforms.RandomResizedCrop(size=size, scale=(0.6, 1))

def get_random_blur(s=6.0):
    return transforms.GaussianBlur(kernel_size=(25,25), sigma=(s, s))

def get_sobel(img):
    img = tf.rgb_to_grayscale(img, 1)
    h_mask = torch.tensor([[1., 0. , -1.], [2., 0., -2.], [1., 0 , -1.]]).unsqueeze(0).unsqueeze(0)
    v_mask = torch.tensor([[1., 2. , 1.], [0., 0., 0.], [-1., -2. , -1.]]).unsqueeze(0).unsqueeze(0)
    sx = F.conv2d(img.squeeze(0), h_mask, stride=1, padding=1)
    sy = F.conv2d(img.squeeze(0), v_mask, stride=1, padding=1)
    sobel = torch.sqrt(torch.pow(sx, 2) + torch.pow(sy, 2)).unsqueeze(0)
    return sobel.repeat(1, 3, 1, 1)

def get_random_flip(prob=0.5):
    return transforms.RandomHorizontalFlip(p=prob)

def get_random_invert(prob=0.5):
    return transforms.RandomInvert(p=prob)

def get_random_rotation(deg=(-90, 90)):
    return transforms.RandomRotation(degrees=deg)

def gray():
    return transforms.Grayscale(num_output_channels=3)