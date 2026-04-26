import json
import os
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
import pdb
import numpy as np
from pathlib import Path

class COCODataSet(Dataset):
    def __init__(self, data_path, trans, coco=None, seg=None, model=None):
        self.data_path = data_path
        self.trans = trans

        img_files = os.listdir(self.data_path)
        random.shuffle(img_files)
        self.img_files = img_files

        self.coco = coco
        self.seg = seg
        self.model = model


    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_file = self.img_files[index]

        img_id = int(img_file.split(".jpg")[0][-6:])

        raw_image = Image.open(os.path.join(self.data_path, img_file)).convert("RGB")

        raw_shape = np.array(raw_image).shape

        if 'deepseek' in  self.model :
            # image = os.path.join(self.data_path, img_file)
            image = self.trans([raw_image])
        else:
            image = self.trans(raw_image)

        return {"img_id": img_id, "image": image, 'raw_image':os.path.join(self.data_path, img_file), 'img_file': img_file}
    

class POPEDataSet(Dataset):
    def __init__(self, pope_path, data_path, trans, model=None):
        self.pope_path = pope_path
        self.data_path = data_path
        self.trans = trans
        self.model = model


        image_list, query_list, label_list = [], [], []


        for q in open(pope_path, 'r'):
            line = json.loads(q)
            image_list.append(line['image'])
            # query_list.append(line['text'] + ' Please answer Yes or No.')
            query_list.append(line['text'])
            label_list.append(line['label'])

        for i in range(len(label_list)):
            if label_list[i] == 'no':
                label_list[i] = 0
            else:
                label_list[i] = 1

        assert len(image_list) == len(query_list)
        assert len(image_list) == len(label_list)

        self.image_list = image_list
        self.query_list = query_list
        self.label_list = label_list

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        image_path = os.path.join(self.data_path, self.image_list[index])
        raw_image = Image.open(image_path).convert("RGB")
       
        if 'deepseek' in  self.model :
            # image = os.path.join(self.data_path, img_file)
            image = self.trans([raw_image])
        else:
            image = self.trans(raw_image)

        query = self.query_list[index]
        label = self.label_list[index]


        return {"image": image, "query": query, "label": label, 'file': self.image_list[index], 'raw_image':image_path}

class MMEDataSet(Dataset):
    def __init__(self, mme_path, mme_type, data_path, trans, model=None):
        self.pope_path = mme_path
        self.mme_type = mme_type
        self.data_path = data_path
        self.trans = trans
        self.model = model

        image_list, query_list, label_list = [], [], []


        for q in open(mme_path, 'r'):
            line = json.loads(q)
            image_list.append(line['image'])
            query_list.append(line['text'])
            label_list.append(line['label'])

        assert len(image_list) == len(query_list)
        assert len(image_list) == len(label_list)

        self.image_list = image_list
        self.query_list = query_list
        self.label_list = label_list

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        image_path = os.path.join(self.data_path, self.mme_type, self.image_list[index])

        raw_image = image_path
        if 'deepseek' in  self.model :
            image = raw_image
        else:
            raw_image = Image.open(image_path).convert("RGB")
            image = self.trans(raw_image)

        query = self.query_list[index]
        label = self.label_list[index]

        return {"image": image, "query": query, "label": label, 'file': self.image_list[index], 'raw_image':image_path}
   
class AMBERDataSet(Dataset):
    def __init__(self, amber_path, amber_type, data_path, trans, model=None):
        self.amber_path = amber_path
        self.amber_type = amber_type
        self.data_path = data_path
        self.trans = trans
        self.model = model

        image_list, query_list, id_list = [], [], []

        p = Path(amber_path)
        with p.open('r', encoding='utf-8') as f:
            data = json.load(f)

        for i, item in enumerate(data):
            image_list.append(item['image'])
            if 'd' in amber_type: 
                query_list.append(item['query'] + ' Please answer Yes or No.')
            else:
                query_list.append(item['query'])
            id_list.append(item['id'])

      
        assert len(image_list) == len(query_list)
        assert len(image_list) == len(id_list)

        self.image_list = image_list
        self.query_list = query_list
        self.id_list = id_list

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        image_path = os.path.join(self.data_path,  self.image_list[index])
        raw_image = Image.open(image_path).convert("RGB")

        if 'deepseek' in  self.model :
            image = self.trans([raw_image])
        else:
            image = self.trans(raw_image)

        query = self.query_list[index]
        id = self.id_list[index]


        return {"id": id, "query": query, "image": image, 'file': self.image_list[index], 'raw_image': image_path}
   

    def __init__(self, pope_path, data_path, trans):
        self.pope_path = pope_path
        self.data_path = data_path
        self.trans = trans

        image_list, query_list, label_list = [], [], []

        for q in open(pope_path, "r"):
            line = json.loads(q)
            image_list.append(line["image"])
            query_list.append(line["text"])
            label_list.append(line["label"])

        for i in range(len(label_list)):
            for j in range(len(label_list[i])):
                if label_list[i][j] == "no":
                    label_list[i][j] = 0
                else:
                    label_list[i][j] = 1

        assert len(image_list) == len(query_list)
        assert len(image_list) == len(label_list)

        self.image_list = image_list
        self.query_list = query_list
        self.label_list = label_list

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        image_path = os.path.join(self.data_path, self.image_list[index])
        raw_image = Image.open(image_path).convert("RGB")
        image = self.trans(raw_image)
        query = self.query_list[index]
        label = self.label_list[index]

        return {"image": image, "query": query, "label": label}