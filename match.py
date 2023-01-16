import time
import torch
import torchvision.models as models
import os, cv2
from PIL import Image, ImageFilter
import math
import numpy as np
#import pandas as pd
import torch.nn as nn
import torchvision.transforms.functional as tf
gpu = True

# 生成模型
model_name = 'vgg'
neg1to1 = False

model = models.maxvit_t('IMAGENET1K_V1')
model.load_state_dict(torch.load('weights/weights_1-15_39.pt'))

if gpu:
    model = model.to(device="cuda")

model.eval()
source = './eval'
target = './target'
result = []
'''
def readImg(name):
    img = cv2.imread(name, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = cv2.filter
    img = cv2.resize(img, (224,224)).transpose([2, 1, 0])
    img = torch.tensor(img.astype(np.float32)).unsqueeze(0)
    return img/255
'''
def readImg(path):
        img = Image.open(path)
        img = img.convert('RGB')
        img = img.filter(ImageFilter.GaussianBlur(radius=2))
        img = img.filter(ImageFilter.EMBOSS)
        img = img.resize((224,224))
        img = tf.to_tensor(img)
        img = tf.normalize(img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)).unsqueeze(0)
        return img

'''
def readImg(name):
    img = cv2.imread(name, cv2.IMREAD_COLOR)#cv2.imdecode(np.fromfile(name, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print('error in loading image', name)
        return None    
    img = cv2.resize(img.astype(np.float32), (224, 224))
    img = torch.from_numpy(img.transpose([2, 1, 0])).unsqueeze(0)
    if neg1to1:
        return img/255 * 2  - 1
    else:
        return img/255
'''

def featurize_images(folder, model):
    fImg_name = []
    img_f = []

    model.eval()
    for fRoot, fDirs, fFiles in os.walk(folder):
        for ffile in fFiles:
            fImg = readImg(os.path.join(fRoot, ffile))
            if fImg is None:
                continue
            fImg_name.append(os.path.join(fRoot, ffile))
            with torch.no_grad():
                fImg = fImg.to(device="cuda")
                f_vec = nn.functional.normalize(model(fImg))

                img_f.append(f_vec)
    
    img_f_mat = torch.cat(img_f, dim=0)
    return img_f_mat, fImg_name



def cos_similarity(f_mat_src, f_mat_target):
    score = torch.matmul(f_mat_src, f_mat_target.T) #cosine similarity of feature vectors
    return score


batch_size = 256

source_name = []
source_f = []
#s_time = time.time()
'''
source_f_mat = []
target_f_mat = []
sscore = []
for i in range(3):
    match model_name:
        case 'vgg': model = models.vgg16(weights='DEFAULT')
        case 'resnet': model = models.resnet50(weights='DEFAULT')
        case 'inception': model = models.inception_v3(weights='Default')
    
    model = model.to(device='cuda')

    source_f_mat[i], source_name = featurize_images(source, model)
    target_f_mat[i], target_name = featurize_images(target, model)

    sscore[i] = similarity_score(source_f_mat[i], target_f_mat[i])
    max_score[i], max_arg[i] = torch.topk(sscore[i], k=5, dim=1)

'''


for sRoot, sDirs, sFiles in os.walk(source):
    for sfile in sFiles:
        sImg = readImg(os.path.join(sRoot, sfile))
        if sImg is None:
            continue
        source_name.append(os.path.join(sRoot, sfile))
        with torch.no_grad():
            sImg = sImg.to(device="cuda")
            f_vec3 = nn.functional.normalize(model(sImg))

            source_f.append(f_vec3)

source_f_mat = torch.cat(source_f, dim=0)

target_name = []
target_f = []

for tRoot, tDirs, tFiles in os.walk(target):
    for tfile in tFiles:
        tImg = readImg(os.path.join(tRoot, tfile))
        if tImg is None:
            continue
        target_name.append(os.path.join(tRoot, tfile))
        with torch.no_grad():
            tImg = tImg.to(device="cuda")
            f_vec3 = nn.functional.normalize(model(tImg))
            target_f.append(f_vec3)


target_f_mat = torch.cat(target_f, dim=0)

score_mat = cos_similarity(source_f_mat, target_f_mat)

max_score, max_arg = torch.topk(score_mat, k=15, dim=1)

#max_score = torch.cat((max_score1, max_score2, max_score3), dim=1)
#max_arg = torch.cat((max_arg1, max_arg2, max_arg3), dim=1)


for i, ip in enumerate(max_arg):
    print('For Image {}:'.format(source_name[i]))
    for k in range(15):
        print('{}'.format(target_name[ip[k]]))


'''
for sRoot, sDirs, sFiles in os.walk(source):
    for sfile in sFiles: # 遍历source下的每张图
        sImg = readImg(os.path.join(sRoot, sfile))
        if sImg is None:
            continue
        source_name.append(os.path.join(sRoot, sfile))
        source_img.append(sImg)

source_img = torch.cat(source_img, dim=0)
if gpu:
    source_img = source_img.cuda()
print('source img shape', source_img.shape, len(source_name))
num_batch = math.ceil(len(source_img) / float(batch_size))
with torch.no_grad():
    for i in range(num_batch):
        start = batch_size * i
        end = min(batch_size * (i + 1), len(source_img))
        tmp_f = model(source_img[start: end]).clone().detach()
        source_f.append(tmp_f)
source_f = torch.cat(source_f, dim=0)
print('source feature shape', source_f.shape, time.time() - s_time)
source_img = None


target_name = []
target_img = []
target_f = []
s_time = time.time()
for tRoot, tDirs, tFiles in os.walk(target):
    for tfile in tFiles:
        tImg = readImg(os.path.join(tRoot, tfile))
        if tImg is None:
            continue
        target_name.append(os.path.join(tRoot, tfile))
        target_img.append(tImg)

target_img = torch.cat(target_img, dim=0)
if gpu:
    target_img = target_img.cuda()
print('target img shape', target_img.shape, len(target_img))
num_batch = math.ceil(len(target_img) / float(batch_size))
with torch.no_grad():
    for i in range(num_batch):
        start = batch_size * i
        end = min(batch_size * (i + 1), len(target_img))
        tmp_f = model(target_img[start: end]).clone().detach()
        target_f.append(tmp_f)
target_f = torch.cat(target_f, dim=0)
print('target feature shape', target_f.shape, time.time() - s_time)
target_img = None

results = []
s_time = time.time()
for i in range(len(source_f)):
    d = [source_name[i]]
    sf = source_f[i: i+1]
    if len(sf.shape) > 2:
        mse = torch.sum((sf - target_f) ** 2, dim=[1,2,3])
    else:
        mse = torch.sum((sf - target_f) ** 2, dim=-1)
    idx = np.argsort(mse.detach().cpu().numpy())
    for j in range(12):
        d.append(target_name[idx[j]])
    results.append(d)
    if i % 100 == 0 or i == len(source_f) - 1:
        print(i, time.time() - s_time)
        df = pd.DataFrame(results)
        if neg1to1:
            df.to_csv('result_{}_neg1to1.csv'.format(model_name))
        else:
            df.to_csv('result_{}.csv'.format(model_name))
'''