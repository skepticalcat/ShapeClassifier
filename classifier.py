from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import ShapeDataset


transform = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize((0.5), (0.5))])

ds = ShapeDataset(transform)
datasets = ds.train_test_dataset(0.2, 0.1)

print("{}, {}, {} Train, Test and Val Samples".format(len(datasets['train']),
                                                      len(datasets['test']),
                                                      len(datasets['val'])))

dataloaders = {x: DataLoader(datasets[x], 8, shuffle=True, num_workers=0, drop_last=True) for x in
               ['train', 'test', "val"]}

for elem in dataloaders["test"]:
    print(elem)