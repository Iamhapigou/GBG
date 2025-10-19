import numpy as np
import os
import torch
from collections import defaultdict
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as TF

def read_data(dataset, idx, is_train=True):
    #Splice path for easy cross - platform use
    if is_train:
        data_dir = os.path.join('./dataset', dataset, 'train/')
    else:
        data_dir = os.path.join('./dataset', dataset, 'test/')

    file = data_dir + str(idx) + '.npz'
    with open(file, 'rb') as f:
        data = np.load(f, allow_pickle=True)['data'].tolist()
    return data

def read_client_data(dataset, idx, args, is_train=True, few_shot=0):
    data = read_data(dataset, idx, is_train)
    if "News" in dataset:
        data_list = process_text(data)
    else:
        data_list = process_image(data)

    if args.dataset == "Cifar100":
        if "EfficientNet" in args.algorithm or "MobileNetV2" in args.algorithm:
            print("process cifar100")
            data_list = process_Cifar100(data)
    else:
        pass

    #Whether to conduct small sample training
    if is_train and few_shot > 0:
        shot_cnt_dict = defaultdict(int)
        data_list_new = []
        for data_item in data_list:
            label = data_item[1].item()
            if shot_cnt_dict[label] < few_shot:
                data_list_new.append(data_item)
                shot_cnt_dict[label] += 1
        data_list = data_list_new

    return data_list

def process_Cifar100(data):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408),
                             std=(0.2675, 0.2565, 0.2761))
    ])
    X = data['x']
    y = data['y']

    processed_data = []

    for i in range(len(X)):
        arr = X[i]

        t = torch.from_numpy(arr)
        if t.dtype != torch.float32:
            t = t.float()
        if t.max().item() > 1.5:
            t = t / 255.0

        pil_img = TF.to_pil_image(t)
        img = transform(pil_img)
        label = int(y[i])

        processed_data.append((img, torch.tensor(label, dtype=torch.long)))

    return processed_data

def process_image(data):
    X = torch.Tensor(data['x']).type(torch.float32)
    y = torch.Tensor(data['y']).type(torch.int64)
    return [(x, y) for x, y in zip(X, y)]

def process_text(data):
    X, X_lens = list(zip(*data['x']))
    y = data['y']
    X = torch.Tensor(X).type(torch.int64)
    X_lens = torch.Tensor(X_lens).type(torch.int64)
    y = torch.Tensor(data['y']).type(torch.int64)
    return [((x, lens), y) for x, lens, y in zip(X, X_lens, y)]


