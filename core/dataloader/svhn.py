import os
import torch
from torchvision import datasets, transforms

def get(data_dir='./dataset/', seed=0):
    data = {}
    taskcla = []
    size = [3, 32, 32]

    svhn_dir = os.path.join(data_dir, 'svhn')
    if not os.path.isdir(svhn_dir):
        os.makedirs(svhn_dir)

    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    train_set = datasets.SVHN(svhn_dir, split='train', download=True, transform=transform)
    test_set = datasets.SVHN(svhn_dir, split='test', download=True, transform=transform)

    data[0] = {}
    data[0]['name'] = 'svhn'
    data[0]['ncla'] = 10
    data[0]['train'] = {'x': [], 'y': []}
    data[0]['test'] = {'x': [], 'y': []}

    for image, target in train_set:
        data[0]['train']['x'].append(image)
        data[0]['train']['y'].append(target)

    for image, target in test_set:
        data[0]['test']['x'].append(image)
        data[0]['test']['y'].append(target)

    data[0]['train']['x'] = torch.stack(data[0]['train']['x']).view(-1, size[0], size[1], size[2])
    data[0]['train']['y'] = torch.LongTensor(data[0]['train']['y']).view(-1)
    data[0]['test']['x'] = torch.stack(data[0]['test']['x']).view(-1, size[0], size[1], size[2])
    data[0]['test']['y'] = torch.LongTensor(data[0]['test']['y']).view(-1)

    taskcla.append((0, data[0]['ncla']))
    return data, taskcla, size
