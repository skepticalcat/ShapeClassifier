import pickle

import torch
from PIL import Image
from torch.utils.data import Dataset


class ShapeDataset(Dataset):

    def __init__(self, transform):
        with open("data.pickle", "rb") as handle:
            self.examples = pickle.load(handle)
        self.transform = transform
        self.classes = {'pentagon': 0,
                        'circle': 1,
                        'nonagon': 2,
                        'triangle': 3,
                        'octagon': 4,
                        'square': 5,
                        'heptagon': 6,
                        'hexagon': 7,
                        'star': 8}

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = Image.fromarray(self.examples[idx][1], 'L')
        tensor_image = self.transform(image)
        torch.set_printoptions(profile="full")
        sample = {"label": self.classes[self.examples[idx][0]], "picture": tensor_image}

        return sample

    def train_test_dataset(self, test_split=0.25, val_split=0.1):
        train_size = 1-test_split-val_split
        subsets = torch.utils.data.random_split(self, [int(len(self)*train_size),int(len(self)*test_split),int(len(self)*val_split)])
        datasets = {'train': subsets[0],
                    'test': subsets[1],
                    'val':  subsets[2]}
        return datasets