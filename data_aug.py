from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_random_color_distortion(s=1.0):
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    return transforms.RandomApply([color_jitter], p=0.8)
    
def get_random_crop():
    return transforms.RandomResizedCrop(size=(224,224), scale=(0.6, 1))

def get_random_blur():
    s = 0
    return transforms.GaussianBlur(kernel_size=(22,22), sigma=(s, s))

def get_sobel(img):
    h_mask = torch.tensor([[1., 2. , 1.], [0., 0., 0.], [-1., -2. , -1.]])
    v_mask = torch.tensor([[-1., 0. , 1.], [-2., 0., 2.], [-1., -2. , 1.]])
    sx = F.conv2d(img.unsqueeze(0), h_mask, stride=1, padding=1)
    sy = F.conv2d(img.unsqueeze(0), v_mask, stride=1, padding=1)
    return torch.sqrt(torch.pow(sx, 2) + torch.pow(sy, 2)).squeeze(0)

def get_random_flip(prob=0.5):
    return transforms.RandomHorizontalFlip(p=prob)

def get_random_invert(prob=0.5):
    return transforms.RandomInvert(p=prob)

def get_random_rotation(deg=(-50, 50)):
    return transforms.RandomRotation(degrees=deg)

def gray():
    return transforms.Grayscale(num_output_channels=3)