import random, os
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms.functional as tf
import numpy as np
from data_aug import *



class DataLoaderCombo:
    def __init__(self, source, target, batch_size, augments):
        self.source_path = source
        self.target_path = target
        self.batch_size = batch_size
        self.index = 0

        self.src = ['./{}/{}'.format(source, img) for path, subdirs, imgs in os.walk(source) for img in imgs]
        self.src += ['./{}/{}'.format(target, img) for path, subdirs, imgs in os.walk(target) for img in imgs]

        self.augments = augments
        random.shuffle(self.src)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == -1:
            raise StopIteration
        if self.index + self.batch_size >= len(self.src):
            batch = self.get_batch(self.src[self.index :])
            self.index = -1
        else:
            batch = self.get_batch(self.src[self.index : self.index + self.batch_size])
            self.index += self.batch_size
        return batch

    def readImg(self, path):
        img = Image.open(path)
        img = img.convert('RGB')
        img = img.resize((224,224))
        img = tf.to_tensor(img)
        img = tf.normalize(img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)).unsqueeze(0)
        return img

    def get_aug(self, img):
        augment = nn.Sequential(
            get_random_color_distortion(),
            get_random_crop((224,224)),
            get_random_flip(),
            #gray(),
        )

        return augment(img)
    
    def get_batch(self, source_paths):
        input_imgs, labels = [], []

        #correct_paths = ['./{}/{}'.format(self.target_path, img_name) for img_name in img_paths]
        
        labels = range(len(source_paths))
        labels = [l + len(source_paths) for l in labels]
        labels += range(len(source_paths))

        if self.augments:
            batch_paths = source_paths 
        else:
            batch_paths = source_paths #+ correct_paths

        augments = []

        for path in batch_paths:
            img = self.readImg(path)
            input_imgs.append(img)
            if self.augments:
                augments.append(self.get_aug(img))
        
        if self.augments:
            input_imgs += augments

        return torch.cat(input_imgs, dim=0).to('cuda'), labels
    