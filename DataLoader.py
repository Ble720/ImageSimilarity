import random, os, cv2, torch
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
    
    def create_pairs(self):
        pairs = []
        for sRoot, sDirs, sFiles in os.walk(self.source_path):
            for sfile in sFiles:
                correct_path = './target/{}.png'.format(sfile[:-4])
                pairs.append(os.path.join(sRoot, sfile).replace('\\', '/'), correct_path, 1)
                random_target = random.choices([os.path.join('target', p) for p in os.listdir(self.target_path) if p != sfile], k=self.n_false)

                for path in random_target:
                    pairs.append(os.path.join(sRoot, sfile).replace('\\', '/'), './target/' + path, 0)

        random.shuffle(pairs)
        self.data = pairs

    def readImg(self, name):
        img = cv2.imread(name, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = cv2.filter
        img = cv2.resize(img, (224,224)).transpose([2, 1, 0])
        img = torch.tensor(img.astype(np.float64)).unsqueeze(0)
        return img/255

    def getImgLblBatch(self, pairs):
        img_1, img_2, labels = [], [], []
        for path_1, path_2, label in pairs:
            img_1.append(self.readImg(path_1))
            img_2.append(self.readImg(path_2))
            labels.append(label)

        return torch.cat(img_1, dim=0), torch.cat(img_2, dim=0), torch.tensor(label)

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

    