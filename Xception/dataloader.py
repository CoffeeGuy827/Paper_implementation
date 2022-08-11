from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import os

path2data = '../data'

# if not exists the path, make the path
if not os.path.exists(path2data):
    os.mkdir(path2data)

data_transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(299)])
train_ds = datasets.STL10(path2data, split='train', download='True', transform=data_transformer)
val_ds = datasets.STL10(path2data,split='test', download='True', transform=data_transformer)

print(train_ds.data.shape)

train_dl = DataLoader(train_ds, batch_size=32, shuffle=32)
val_dl = DataLoader(val_ds, batch_size=32, shuffle=32)