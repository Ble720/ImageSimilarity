from PIL import Image, ImageFilter
import imagehash
import os
import numpy as np

img_hashes = []
img_names = []

for subdir, dirs, origImg in os.walk('./Original'):

    for img in origImg:
        im = Image.open('./Original/' + img)
        im = im.filter(ImageFilter.EMBOSS)
        #im = im.filter(ImageFilter.MaxFilter(size=7))
        
        img_hash  = imagehash.whash(im)
        img_hashes.append(img_hash)
        img_names.append(img)

crop_hashes = []
crop_names = []


for subdir, dirs, crops in os.walk('./Crops'):
    for crop in crops:
        crop_img = Image.open('./Crops/' + crop)
        crop_img = crop_img.filter(ImageFilter.EMBOSS)
        crop_hash = imagehash.whash(crop_img)
        
        crop_hashes.append(crop_hash)
        crop_names.append(crop)


hash_distances = np.tile(img_hashes, (len(crop_hashes), 1)).T
hash_distances = np.absolute(hash_distances - np.array(crop_hashes))
best_similarity = np.argmin(hash_distances, axis=1).flatten()

for c, i in enumerate(best_similarity):
    print('For Image {}: {}'.format(img_names[c], crop_names[i]))