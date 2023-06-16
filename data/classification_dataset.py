import os
import json
import pandas as pd

from torch.utils.data import Dataset
from PIL import Image

from data.utils import pre_caption


class ClassificationDataset(Dataset):
    def __init__(self, transform, ann_root, split):
        '''
        image_root (string): Root directory of images 
        ann_root (string): directory to store the annotation file
        split (string): train, val or test
        '''
        filenames = {'train': 'train.csv', 'val': 'dev.csv', 'test': 'test.csv'}

        self.annotation = pd.read_csv(os.path.join(ann_root, filenames[split]))
        self.annotation = self.annotation.reset_index()
        self.transform = transform

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        ann = self.annotation.loc[index]

        image_path = os.path.join(ann['local_path'])
        try:
            image = Image.open(image_path).convert('RGB')
        except OSError:
            return self.__getitem__(index-1)
        image = self.transform(image)

        sentence = ann['text']

        label = int(ann['label'])

        return image, sentence, label
