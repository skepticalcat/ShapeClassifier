import pickle

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset


class ShapeDataset(Dataset):

    def __init__(self, transform):
        with open("data.pickle", "rb") as handle:
            self.examples = pickle.load(handle)
        self.transform = transform

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = Image.fromarray(self.examples[idx][1],'L')
        tensor_image = self.transform(image)
        sample = {"labels": self.examples[idx][0], "picture": tensor_image}
        return sample

    def train_test_dataset(self, test_split=0.25, val_split=0.1):
        train_idx, test_idx = train_test_split(list(range(len(self))), test_size=test_split)
        train_idx, val_idx = train_test_split(list(range(len(train_idx))), test_size=val_split)
        datasets = {'train': Subset(self, train_idx),
                    'test': Subset(self, test_idx),
                    'val': Subset(self, val_idx)}
        return datasets