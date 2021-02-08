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
        tensor_image = torch.from_numpy(self.examples[idx][1]).float()
        tensor_image.unsqueeze_(0)
        #tensor_image = self.transform(tensor_image)
        #torch.set_printoptions(profile="full")
        #print(tensor_image)
        sample = {"label": self.classes[self.examples[idx][0]], "picture": tensor_image}
        return sample

    def train_test_dataset(self, test_split=0.25, val_split=0.1):
        train_idx, test_idx = train_test_split(list(range(len(self))), test_size=test_split)
        train_idx, val_idx = train_test_split(list(range(len(train_idx))), test_size=val_split)
        datasets = {'train': Subset(self, train_idx),
                    'test': Subset(self, test_idx),
                    'val': Subset(self, val_idx)}
        return datasets