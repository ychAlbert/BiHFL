from __future__ import print_function

import os
import sys
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import urllib.request
import zipfile
import shutil

class TinyImageNet(Dataset):
    def __init__(self, root, train):
        super(TinyImageNet, self).__init__()
        self.root = os.path.join(root, 'tiny-imagenet-200')
        self.train = train
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        # 下载并解压数据集
        if not os.path.exists(self.root):
            self._download_and_extract()

        # 加载数据
        self.data = []
        self.labels = []
        self._load_data()

    def _download_and_extract(self):
        url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
        zip_path = os.path.join(self.root, '..', 'tiny-imagenet-200.zip')
        os.makedirs(os.path.dirname(zip_path), exist_ok=True)

        print('Downloading Tiny ImageNet...')
        urllib.request.urlretrieve(url, zip_path)

        print('Extracting Tiny ImageNet...')
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(self.root))

        os.remove(zip_path)

    def _load_data(self):
        if self.train:
            data_dir = os.path.join(self.root, 'train')
            classes = os.listdir(data_dir)
            for class_idx, class_name in enumerate(classes):
                class_dir = os.path.join(data_dir, class_name, 'images')
                if os.path.exists(class_dir):
                    for img_name in os.listdir(class_dir):
                        img_path = os.path.join(class_dir, img_name)
                        self.data.append(img_path)
                        self.labels.append(class_idx)
        else:
            data_dir = os.path.join(self.root, 'val')
            with open(os.path.join(data_dir, 'val_annotations.txt'), 'r') as f:
                for line in f:
                    img_name, class_name, *_ = line.strip().split('\t')
                    img_path = os.path.join(data_dir, 'images', img_name)
                    if os.path.exists(img_path):
                        self.data.append(img_path)
                        self.labels.append(self._get_class_idx(class_name))

    def _get_class_idx(self, class_name):
        class_to_idx_path = os.path.join(self.root, 'wnids.txt')
        if not hasattr(self, 'class_to_idx'):
            self.class_to_idx = {}
            with open(class_to_idx_path, 'r') as f:
                for idx, line in enumerate(f):
                    self.class_to_idx[line.strip()] = idx
        return self.class_to_idx[class_name]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

class iTinyImageNet(TinyImageNet):
    def __init__(self, root, classes, memory_classes, memory, task_num, train, transform=None):
        super(iTinyImageNet, self).__init__(root=root, train=train)

        self.transform = transform
        if not isinstance(classes, list):
            classes = [classes]

        self.class_mapping = {c: i for i, c in enumerate(classes)}
        self.class_indices = {}

        for cls in classes:
            self.class_indices[self.class_mapping[cls]] = []

        data = []
        labels = []
        tt = []  # task module labels
        td = []  # discriminator labels

        for i in range(len(self.data)):
            if self.labels[i] in classes:
                data.append(self.data[i])
                labels.append(self.class_mapping[self.labels[i]])
                tt.append(task_num)
                td.append(task_num + 1)
                self.class_indices[self.class_mapping[self.labels[i]]].append(i)

        if memory_classes:
            for task_id in range(task_num):
                for i in range(len(memory[task_id]['x'])):
                    if memory[task_id]['y'][i] in range(len(memory_classes[task_id])):
                        data.append(memory[task_id]['x'][i])
                        labels.append(memory[task_id]['y'][i])
                        tt.append(memory[task_id]['tt'][i])
                        td.append(memory[task_id]['td'][i])

        self.data = data
        self.labels = labels
        self.tt = tt
        self.td = td

    def __getitem__(self, index):
        img_path = self.data[index]
        target = self.labels[index]
        tt = self.tt[index]
        td = self.td[index]

        if isinstance(img_path, str):
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
        else:
            img = img_path

        return img, target

    def __len__(self):
        return len(self.data)

class DatasetGen(object):
    def __init__(self, data_dir, seed):
        super(DatasetGen, self).__init__()

        self.seed = seed
        self.batch_size = 64
        self.root = data_dir
        self.use_memory = 'yes'

        self.num_tasks = 20
        self.num_classes = 200

        self.num_samples = 0

        self.inputsize = [3, 64, 64]
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        self.transformation = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

        self.taskcla = [[t, int(self.num_classes / self.num_tasks)] for t in range(self.num_tasks)]

        self.indices = {}
        self.dataloaders = {}
        self.idx = {}

        self.num_workers = 4
        self.pin_memory = True

        np.random.seed(self.seed)
        task_ids = np.split(np.random.permutation(self.num_classes), self.num_tasks)
        self.task_ids = [list(arr) for arr in task_ids]

        self.train_set = {}
        self.train_split = {}
        self.test_set = {}

        self.task_memory = {}
        for i in range(self.num_tasks):
            self.task_memory[i] = {}
            self.task_memory[i]['x'] = []
            self.task_memory[i]['y'] = []
            self.task_memory[i]['tt'] = []
            self.task_memory[i]['td'] = []
            
        # 预先下载和准备数据集
        dummy_dataset = TinyImageNet(root=self.root, train=True)
        if not os.path.exists(dummy_dataset.root):
            dummy_dataset._download_and_extract()

    def get(self, task_id):
        self.dataloaders[task_id] = {}

        sys.stdout.flush()

        if task_id == 0:
            memory_classes = None
            memory = None
        else:
            memory_classes = self.task_ids
            memory = self.task_memory

        self.train_set[task_id] = iTinyImageNet(root=self.root, classes=self.task_ids[task_id],
                                                memory_classes=memory_classes, memory=memory,
                                                task_num=task_id, train=True, transform=self.transformation)

        self.test_set[task_id] = iTinyImageNet(root=self.root, classes=self.task_ids[task_id],
                                               memory_classes=None, memory=None,
                                               task_num=task_id, train=False, transform=self.transformation)

        print(len(self.test_set[task_id]))

        train_loader = torch.utils.data.DataLoader(self.train_set[task_id], batch_size=2500,
                                                   pin_memory=self.pin_memory, shuffle=True)
        
        test_loader = torch.utils.data.DataLoader(self.test_set[task_id], batch_size=500,
                                                  pin_memory=self.pin_memory, shuffle=True)

        self.dataloaders[task_id]['train'] = {'x': [], 'y': []}
        for data, label in train_loader:
            self.dataloaders[task_id]['train']['x'] = data
            self.dataloaders[task_id]['train']['y'] = label

        self.dataloaders[task_id]['test'] = {'x': [], 'y': []}
        for data, label in test_loader:
            self.dataloaders[task_id]['test']['x'] = data
            self.dataloaders[task_id]['test']['y'] = label

        self.dataloaders[task_id]['name'] = 'iTinyImageNet-{}-{}'.format(task_id, self.task_ids[task_id])

        print("Task ID: ", task_id)
        print("Training set size:   {} images of {}x{}".format(len(train_loader.dataset), self.inputsize[1],
                                                               self.inputsize[1]))
        print("Test set size:       {} images of {}x{}".format(len(test_loader.dataset), self.inputsize[1],
                                                               self.inputsize[1]))

        if self.use_memory == 'yes' and self.num_samples > 0:
            self.update_memory(task_id)

        return self.dataloaders

    def update_memory(self, task_id):
        num_samples_per_class = self.num_samples // len(self.task_ids[task_id])
        mem_class_mapping = {i: i for i, c in enumerate(self.task_ids[task_id])}

        for i in range(len(self.task_ids[task_id])):
            data_loader = torch.utils.data.DataLoader(self.train_split[task_id], batch_size=1,
                                                      num_workers=self.num_workers,
                                                      pin_memory=self.pin_memory)

            randind = torch.randperm(len(data_loader.dataset))[:num_samples_per_class]

            for ind in randind:
                self.task_memory[task_id]['x'].append(data_loader.dataset[ind][0])
                self.task_memory[task_id]['y'].append(mem_class_mapping[i])
                self.task_memory[task_id]['tt'].append(data_loader.dataset[ind][2])
                self.task_memory[task_id]['td'].append(data_loader.dataset[ind][3])

        print('Memory updated by adding {} images'.format(len(self.task_memory[task_id]['x'])))