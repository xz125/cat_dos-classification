import os
import torch
from torch.utils.data import DataLoader
import argparse
from torchvision import transforms,datasets
from PIL import Image

def readImg(path):
    return Image.open(path)

def ImageDataset(args):
    data_transforms = {
        'train':transforms.Compose([
            transforms.Resize(150),
            transforms.CenterCrop(150),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test':transforms.Compose([
            transforms.Resize(150),
            transforms.CenterCrop(150),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        }
    data_dir = args.data_dir
    image_datasets = {x:datasets.ImageFolder(os.path.join(data_dir,x),
                        data_transforms[x],loader=readImg)
                        for x in ['train','test']}
    dataloaders = {x: DataLoader(image_datasets[x],
                        batch_size=args.batch_size, shuffle=(x == 'train'),
                        num_workers=args.num_workers)
                        for x in ['train', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes
    return dataloaders,dataset_sizes,class_names
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='classification')
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    #dataloders, dataset_sizes, class_names = ImageDataset(args)
    dataloders, dataset_sizes, class_names = ImageDataset(args)
    print(dataloders.__len__())
    print(class_names)
    print(dataset_sizes)