import os
import json
import re
from io import BytesIO

import pandas as pd
import requests as req
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

        if 'local_path' in ann.columns:
            image_path = os.path.join(ann['local_path'])
            try:
                image = Image.open(image_path).convert('RGB')
            except OSError:
                return self.__getitem__(index - 1)
        else:
            url = re.sub("https://\w*?\.meituan.net/", "https://p.vip.sankuai.com/", ann['pic_url'])
            response = req.get(url, stream=True)
            num = 0
            while response.status_code != 200:
                response = req.get(url, stream=True)
                num += 1
                if num == 20:
                    break
            image = Image.open(BytesIO(response.content))
            image = image.convert('RGB') if image.mode != 'RGB' else image
        image = self.transform(image)

        sentence = ann['text']

        label = int(ann['label'])

        return image, sentence, label


class ClassificationEvalDataset(Dataset):
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

        if 'local_path' in ann.columns:
            image_path = os.path.join(ann['local_path'])
            try:
                image = Image.open(image_path).convert('RGB')
            except OSError:
                return self.__getitem__(index - 1)
        else:
            url = re.sub("https://\w*?\.meituan.net/", "https://p.vip.sankuai.com/", ann['pic_url'])
            response = req.get(url, stream=True)
            num = 0
            while response.status_code != 200:
                response = req.get(url, stream=True)
                num += 1
                if num == 20:
                    break
            image = Image.open(BytesIO(response.content))
            image = image.convert('RGB') if image.mode != 'RGB' else image

        image = self.transform(image)

        sentence = ann['text']

        label = int(ann['label'])
        if 'group' in ann:
            group = int(ann['group'])
        else:
            group = 0

        return image, sentence, label, group
