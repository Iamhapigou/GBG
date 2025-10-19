import os
import ujson
import numpy as np
import gc
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from PIL import Image
import re
import json
from collections import  Counter
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

batch_size = 10
train_ratio = 0.95  # merge original training set and test set, then split it manually.
alpha = 1  # for Dirichlet distribution. 100 for exdir

ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
NUM_LETTERS = len(ALL_LETTERS)

def check(config_path, train_path, test_path, num_clients, niid=False,
          balance=True, partition=None):
    # check existing dataset
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = ujson.load(f)
        if config['num_clients'] == num_clients and \
                config['non_iid'] == niid and \
                config['balance'] == balance and \
                config['partition'] == partition and \
                config['alpha'] == alpha and \
                config['batch_size'] == batch_size:
            print("\nDataset already generated.\n")
            return True

    dir_path = os.path.abspath(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.abspath(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return False

def separate_data(data, num_clients, num_classes, niid=False, balance=False, partition=None, class_per_client=None):
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]

    dataset_content, dataset_label = data
    # guarantee that each client must have at least one batch of data for testing.
    least_samples = int(min(batch_size / (1 - train_ratio), len(dataset_label) / num_clients / 2))

    dataidx_map = {}

    #iid
    if not niid:
        partition = 'pat'
        class_per_client = num_classes

    if partition == 'pat':
        idxs = np.array(range(len(dataset_label)))
        idx_for_each_class = []
        #Sort the data in order for each class
        for i in range(num_classes):
            idx_for_each_class.append(idxs[dataset_label == i])

        # Assign each class to the client separately
        class_num_per_client = [class_per_client for _ in range(num_clients)]
        for i in range(num_classes):
            selected_clients = []
            for client in range(num_clients):
                if class_num_per_client[client] > 0:
                    selected_clients.append(client)
            if len(selected_clients) == 0:
                break
            selected_clients = selected_clients[:int(np.ceil((num_clients / num_classes) * class_per_client))]

            num_all_samples = len(idx_for_each_class[i])
            num_selected_clients = len(selected_clients)
            num_per = num_all_samples / num_selected_clients
            if balance:
                num_samples = [int(num_per) for _ in range(num_selected_clients - 1)]
            else:
                num_samples = np.random.randint(max(num_per / 10, least_samples / num_classes), num_per,
                                                num_selected_clients - 1).tolist()
            num_samples.append(num_all_samples - sum(num_samples))

            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                if client not in dataidx_map.keys():
                    dataidx_map[client] = idx_for_each_class[i][idx:idx + num_sample]
                else:
                    dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx + num_sample],
                                                    axis=0)
                idx += num_sample
                class_num_per_client[client] -= 1

    #Dirichlet distribution
    elif partition == "dir":
        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
        min_size = 0
        K = num_classes
        N = len(dataset_label)

        try_cnt = 1
        while min_size < least_samples:
            if try_cnt > 1:
                print(
                    f'Client data size does not meet the minimum requirement {least_samples}. Try allocating again for the {try_cnt}-th time.')

            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(dataset_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            try_cnt += 1

        for j in range(num_clients):
            dataidx_map[j] = idx_batch[j]

    else:
        raise NotImplementedError

    # assign data
    for client in range(num_clients):
        idxs = dataidx_map[client]
        X[client] = dataset_content[idxs]
        y[client] = dataset_label[idxs]

        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client] == i))))

    del data
    # gc.collect()

    for client in range(num_clients):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)

    return X, y, statistic


def split_data(X, y):
    # Split dataset, ratio: 3:1 train:test
    train_data, test_data = [], []
    num_samples = {'train': [], 'test': []}

    # X[i], y[i] are the class samples
    for i in range(len(y)):
        X_train, X_test, y_train, y_test = train_test_split(
            X[i], y[i], train_size=train_ratio, shuffle=True)

        train_data.append({'x': X_train, 'y': y_train})
        num_samples['train'].append(len(y_train))
        test_data.append({'x': X_test, 'y': y_test})
        num_samples['test'].append(len(y_test))

    print("Total number of samples:", sum(num_samples['train'] + num_samples['test']))
    print("The number of train samples:", num_samples['train'])
    print("The number of test samples:", num_samples['test'])
    print()
    del X, y
    # gc.collect()

    return train_data, test_data


def save_file(config_path, train_path, test_path, train_data, test_data, num_clients,
              num_classes, statistic, niid=False, balance=True, partition=None):
    config = {
        'num_clients': num_clients,
        'num_classes': num_classes,
        'non_iid': niid,
        'balance': balance,
        'partition': partition,
        'Size of samples for labels in clients': statistic,
        'alpha': alpha,
        'batch_size': batch_size,
    }

    # gc.collect()
    print("Saving to disk.\n")

    for idx, train_dict in enumerate(train_data):
        with open(os.path.join(train_path, str(idx) + '.npz'), 'wb') as f:
            np.savez_compressed(f, data=train_dict)
    for idx, test_dict in enumerate(test_data):
        with open(os.path.join(test_path, str(idx) + '.npz'), 'wb') as f:
            np.savez_compressed(f, data=test_dict)
    with open(config_path, 'w') as f:
        ujson.dump(config, f)

    print("Finish generating dataset.\n")

def tokenizer(text, max_len, max_tokens=32000):
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(
        map(tokenizer, iter(text)),
        specials = ['<pad>', 'cls', '<unk>', '<eos>'],
        special_first = True,
        max_tokens = max_tokens
    )
    vocab.set_default_index(vocab['<unk>'])
    text_pipeline = lambda x: vocab(tokenizer(x))

    text_list = []
    for t in text:
        tokens = [vocab['<cls>']] + text_pipeline(t)
        padding = [0 for i in range(max_len - len(tokens))]
        tokens.extend(padding)
        text_list.append(tokens[:max_len])
    return vocab, text_list

def letter_to_index(letter):
    '''returns one-hot representation of given letter
    '''
    index = ALL_LETTERS.find(letter)
    return index

def _one_hot(index, size):
    '''returns one-hot vector with given size and value 1 at given index
    '''
    vec = [0 for _ in range(size)]
    vec[int(index)] = 1
    return vec

def letter_to_vec(letter):
    '''returns one-hot representation of given letter
    '''
    index = ALL_LETTERS.find(letter)
    return _one_hot(index, NUM_LETTERS)

def word_to_indices(word):
    '''returns a list of character indices

    Args:
        word: string

    Return:
        indices: int list with length len(word)
    '''
    print('num of letters (classes):', NUM_LETTERS)
    indices = []
    for c in word:
        indices.append(ALL_LETTERS.find(c))
    return indices


class ImageDataset(Dataset):
    def __init__(self, dataframe, image_folder, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing file names
            image_folder (str): Path to the folder containing the images
            transform (callable, optional): Optional transform to be applied to the image
        """
        self.dataframe = dataframe
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get the file name from the DataFrame
        img_name = self.dataframe.iloc[idx]['file_name']
        img_label = self.dataframe.iloc[idx]['class']
        img_path = os.path.join(self.image_folder, img_name)

        # Load the image using PIL
        image = Image.open(img_path).convert('RGB')  # Ensure RGB if not grayscale

        if self.transform:
            image = self.transform(image)

        return image, img_label
