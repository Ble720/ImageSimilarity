import random, os
from PIL import Image, ImageFilter
import torch
import torch.nn as nn
import torchvision.transforms.functional as tf
import numpy as np
from data_aug import *



class DataLoader:
    def __init__(self, source, target, batch_size, mode):
        self.source_path = source
        self.target_path = target
        self.batch_size = batch_size
        self.index = 0
        self.data = []
        self.mode = mode
        self.src = [img for path, subdirs, imgs in os.walk(source) for img in imgs]
        self.augments = []
        random.shuffle(self.src)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == -1:
            raise StopIteration
        
        batch = self.get_batch(self.src[self.index : self.index + self.batch_size/16])

        self.index += 1
        if self.index > len(self.src):
            self.index = -1
        return batch

    def readImg(self, path):
        img = Image.open(path)
        img = img.convert('RGB')
        img = img.resize((224,224))
        img = tf.to_tensor(img)
        img = tf.normalize(img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)).unsqueeze(0)
        return img

    def create_augments(self):
        color_augment = nn.Sequential(
            get_random_crop(),
            gray(),
            get_random_flip(),
        )



        self.augments.append()

    def augmentations(self, image):
        aug_img = []
        color_dist = get_random_color_distortion(s=1)
        g_blur = get_random_blur()

        sobel_img = get_sobel(image)
        color_img = color_dist(image)
        blur_img = g_blur(image)
        
        end_aug = nn.Sequential(
            get_random_crop(),
            gray(),
            get_random_flip(),
        )

        aug_img.append(end_aug(image))
        return aug_img

    def get_batch(self, img_paths):
        input_imgs, labels = [], []

        num_false = self.batch_size/8

        source_paths = ['./source/{}'.format(img_name) for img_name in img_paths]
        correct_paths = ['./target/{}'.format(img_name) for img_name in img_paths]
        wrong_paths = random.choices([os.path.join(path, img).replace('\\', '/') for path, subdirs, imgs in os.walk(self.target_path) for img in imgs if img not in correct_paths], k=num_false)

        pre_labels = list([[x,x,x,x] for x in range(1, int(num_false/2 + 1))])*2 
        labels = [label for sublist in pre_labels for label in sublist] + [0] * num_false * 4

        batch_paths = source_paths + correct_paths + wrong_paths

        for index, path in enumerate(batch_paths):
            img = self.readImg(path)
            input_imgs.append(img)
            input_imgs += self.augmentations(img)

        return torch.cat(input_imgs, dim=0).to('cuda'), torch.tensor(labels).type(torch.float32).to('cuda')
    

    