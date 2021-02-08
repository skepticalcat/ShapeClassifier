import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision import transforms

from TestValidate import TestValidate
from dataset import ShapeDataset
from model import ShapesCNN

transform = transforms.Compose([transforms.Normalize((0.5), (0.5))])

ds = ShapeDataset(transform)
datasets = ds.train_test_dataset(0.2, 0.1)


batch_size = 8
losses = []
running_loss = 0
val_counter = 0
epochs = 4

dataloaders = {x: DataLoader(datasets[x], batch_size, shuffle=True, num_workers=0, drop_last=True) for x in
               ['train', 'test', "val"]}

print("{}, {}, {} Train, Test and Val Samples".format(len(datasets['train']),
                                                      len(datasets['test']),
                                                      len(datasets['val'])))

net = ShapesCNN()
net.cuda()
net.train()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)  # , eps=0.001)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
testval = TestValidate(device,batch_size,criterion,net)

for epoch in range(epochs):
    for i, data in enumerate(dataloaders["train"], 0):
        val_counter += 1

        inputs, labels = data["picture"], data["label"]
        inputs = inputs.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)

        labels = labels.view(batch_size, -1).squeeze(1).long().to(device)
        loss = criterion(outputs.view(batch_size, -1), labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 0:
            losses.append(running_loss / 10)
            print('[%d, %5d] loss: %.3f' % (epoch, i, losses[-1]))
            running_loss = 0.0
        if val_counter % 500 == 0:
            testval.eval(dataloaders["val"])

testval.eval(dataloaders["test"])

