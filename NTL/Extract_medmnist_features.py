
import os
import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from loader.utils import CustomDataset
import numpy as np
import medmnist
from medmnist import INFO, Evaluator

def initialize_model(model_name, use_pretrained=True):
    model_ft = None
    input_size = 0
    if model_name == "resnet152":
        model_ft = models.resnet152(pretrained=use_pretrained)
        input_size = 224
    return model_ft,input_size


def data_transform_color(input_size):
    return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def data_transform_gray(input_size):
    return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def extract_feature(train_data,train_label,test_data,test_label):

    model_ft, input_size = initialize_model('resnet152')
    feature_extractor = torch.nn.Sequential(*list(model_ft.children())[:-1]).to('cuda')
    if len(train_data.shape)==3:
        transform = data_transform_gray(input_size)
    else:
        transform = data_transform_color(input_size)

    trainset = CustomDataset(train_data,train_label,transform=transform)
    testset = CustomDataset(test_data,test_label,transform=transform)

    train_loader = DataLoader(trainset, batch_size=256,shuffle=False)
    test_loader = DataLoader(testset, batch_size=256,shuffle=False)

    train_features = []
    test_features = []
    train_targets = []
    test_targets = []

    feature_extractor.eval()
    with torch.no_grad():
        for data in train_loader:
            samples = data['sample'].to('cuda')
            labels = data['label']
            feature = feature_extractor(samples)
            train_features.append(feature.cpu())
            train_targets.append(labels.squeeze())
        train_features = torch.cat(train_features,0).squeeze()
        train_targets = torch.cat(train_targets,0)
        for data in test_loader:
            samples = data['sample'].to('cuda')
            labels = data['label']
            feature = feature_extractor(samples)
            test_features.append(feature.cpu())
            test_targets.append(labels.squeeze())

        test_features = torch.cat(test_features,0).squeeze()
        test_targets = torch.cat(test_targets,0)

    return [train_features,train_targets],[test_features,test_targets]

if __name__=='__main__':
    root = './DATA/'

    flags = ['path',   'derma', 'oct', 'pneumonia',  'tissue', 'blood', 'organa', 'organs', 'organc']
    for f in flags:
        
        folder = os.path.join(root, f)
        os.makedirs(folder, exist_ok=True)

        dataflag = f'{f}mnist'
        # download data
        info = INFO[dataflag]
        DataClass = getattr(medmnist, info['python_class'])
        DataClass(split='test', download=True, root=folder)

        # extract features
        data = np.load(os.path.join(folder, f'{dataflag}.npz'))
        train_data = data['train_images']
        train_label = data['train_labels']
        test_data = data['test_images']
        test_label = data['test_labels']
        
        trainset, testset =extract_feature(train_data, train_label, test_data, test_label)
        torch.save(trainset, os.path.join(folder, 'trainset_2048.pt'))
        torch.save(testset, os.path.join(folder, 'testset_2048.pt'))