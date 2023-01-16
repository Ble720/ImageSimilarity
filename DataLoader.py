import random, os
from PIL import Image, ImageFilter
import torch
import torch.nn as nn
import torchvision.transforms.functional as tf
import numpy as np



class DataLoader:
    def __init__(self, source, target, num_false, batch_size):
        self.source_path = source
        self.target_path = target
        self.n_false = num_false
        self.batch_size = batch_size
        self.index = 0
        self.data = []
        self.create_pairs()

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == -1:
            raise StopIteration

        end = self.index + self.batch_size
        if end < len(self.data):
            batch = self.data[self.index:end]
            self.index = end
        else:
            batch = self.data[self.index:]
            self.index = -1
            
        return self.getImgLblBatch(batch)
    
    def create_pairs(self):
        pairs = []
        for sRoot, sDirs, sFiles in os.walk(self.source_path):
            for sfile in sFiles:
                add = []
                source = os.path.join(sRoot, sfile).replace('\\', '/')
                correct_path = './target/{}'.format(sfile)
                add.append([source, correct_path, 1])
                random_target = random.choices([os.path.join(path, img).replace('\\', '/') for path, subdirs, imgs in os.walk(self.target_path) for img in imgs if img not in correct_path], k=self.n_false)

                for path in random_target:
                    add.append([source, path, -1])
                random.shuffle(add)
                pairs.append(add)

        random.shuffle(pairs)
        pairs = [ex for batch in pairs for ex in batch]
        self.data = pairs

    def readImg(self, path):
        img = Image.open(path)
        img = img.convert('RGB')
        #img = img.filter(ImageFilter.GaussianBlur(radius=2))
        #img = img.filter(ImageFilter.EMBOSS)
        img = img.resize((224,224))
        img = tf.to_tensor(img)
        img = tf.normalize(img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)).unsqueeze(0)
        return img

    def getImgLblBatch(self, pairs):
        img_1, img_2, labels = [], [], []
        for path_1, path_2, label in pairs:
            img_1.append(self.readImg(path_1))
            img_2.append(self.readImg(path_2))
            labels.append(label)

        return torch.cat(img_1, dim=0).to('cuda'), torch.cat(img_2, dim=0).to('cuda'), torch.tensor(labels).type(torch.float32).to('cuda')# pairs[:-1]

    

    