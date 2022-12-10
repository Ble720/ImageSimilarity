from PIL import Image
import imagehash
import os
import numpy as np

crop_hashes = []
crop_names = []
for subdir, dirs, cropImg in os.walk('./Crops'):

    for crop in cropImg:
        crop_hash  = imagehash.crop_resistant_hash(Image.open('./Crops/' + crop))
        crop_hashes.append(crop_hash)
        crop_names.append(crop)


full_img_hashes = []
img_names = []

for subdir, dirs, img in os.walk('./Original'):
    for full_img in img:
        img_hash = imagehash.crop_resistant_hash(Image.open('./Original/' + full_img))
        
        full_img_hashes.append(img_hash)
        img_names.append(full_img)


hash_distances = np.tile(crop_hashes, (len(full_img_hashes), 1)).T
hash_distances = np.absolute(hash_distances - np.array(full_img_hashes))
best_similarity = np.argmin(hash_distances, axis=1).flatten()

for c, i in enumerate(best_similarity):
    print('For Crop {}: {}'.format(crop_names[c], img_names[i]))