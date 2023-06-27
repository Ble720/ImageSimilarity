import torch
import shutil, os
from PIL import Image

import torch.nn as nn
import torchvision.transforms.functional as tf
from ModelCombo import ModelCombo

import argparse

def readImg(path):
        img = Image.open(path)
        img = img.convert('RGB')
        img = img.resize((224,224))
        img = tf.to_tensor(img)
        img = tf.rgb_to_grayscale(img, num_output_channels=3)
        img = tf.normalize(img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)).unsqueeze(0)
        return img

def featurize_images(folder, model):
    fImg_name = []
    img_f = []

    model.eval()
    for fRoot, fDirs, fFiles in os.walk(folder):
        for ffile in fFiles:
            fImg = readImg(os.path.join(fRoot, ffile))
            if fImg is None:
                continue
            fImg_name.append(os.path.join(fRoot, ffile).replace('\\', '/'))
            with torch.no_grad():
                fImg = fImg.to(device="cuda")
                f_vec = nn.functional.normalize(model(fImg))

                img_f.append(f_vec)
    
    img_f_mat = torch.cat(img_f, dim=0)
    return img_f_mat, fImg_name

def cos_similarity(f_mat_src, f_mat_target):
    score = torch.matmul(f_mat_src, f_mat_target.T) #cosine similarity of feature vectors
    return score

def match(model, source, target, topk, match_name):
    model.eval()

    source_name = []
    source_f = []

    output_folder = match_name != ''

    for sRoot, sDirs, sFiles in os.walk(source):
        for sfile in sFiles:
            sImg = readImg(os.path.join(sRoot, sfile))
            if sImg is None:
                continue
            source_name.append(os.path.join(sRoot, sfile).replace('\\', '/'))
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
            target_name.append(os.path.join(tRoot, tfile).replace('\\', '/'))
            with torch.no_grad():
                tImg = tImg.to(device="cuda")
                f_vec3 = nn.functional.normalize(model(tImg))
                target_f.append(f_vec3)

    target_f_mat = torch.cat(target_f, dim=0)
    score_mat = cos_similarity(source_f_mat, target_f_mat)

    
        
    max_score, max_arg = torch.topk(score_mat, k=topk, dim=1)

    acc_k = []

    for i, ip in enumerate(max_arg):
        passed = False
        #print('For Image {}:'.format(source_name[i]))
        si_name = source_name[i].split('/')[-1]
        
        if output_folder:
            os.makedirs('{}/{}'.format(match_name, i))
            shutil.copyfile(source_name[i], '{}/{}/source_{}'.format(match_name, i, si_name))

        t_names = []
        for k in range(topk):
            ti_name = target_name[ip[k]].split('/')[-1]
            t_names.append(ti_name)

            if output_folder:
                shutil.copyfile(target_name[ip[k]], '{}/{}/{}'.format(match_name, i, ti_name))
            #else:
            #    print(ti_name)
                
            if ti_name == si_name:
                passed = True
                acc_k.append(k+1)
                break
            
        #if passed:
        #    print('Pass')
        #else:
        #  for path in t_names:
        #    print(path)
            
    for k in range(topk):
        count = len([i for i in acc_k if i <= k+1])
        print('Top {} Eval Acc: {}/{}'. format(k+1 ,count, max_arg.shape[0]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights_2-11_199.pt', help='initial weights path')
    parser.add_argument('--topk', type=int, default=15)
    parser.add_argument('--source', type=str, default='./eval')
    parser.add_argument('--target', type=str, default='./target')
    parser.add_argument('--output-folder', type=str, default='')
    opt = parser.parse_args()
    model = ModelCombo()
    model.load_state_dict(torch.load(opt.weights), strict=False)
    gpu = True
    if gpu:
        model = model.to(device="cuda")

    match(model, opt.source, opt.target, opt.topk, opt.output_folder)