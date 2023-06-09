
import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from loader.utils import CustomDataset
import numpy as np

def initialize_model(model_name, use_pretrained=True):
    model_ft = None
    input_size = 0
    if model_name == "resnet152":
        model_ft = models.resnet152(pretrained=use_pretrained)
        input_size = 224
    return model_ft,input_size

def data_transform_color(input_size):
    return transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def data_transform_gray(input_size):
    return transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def extract_cifar_feature(root):
    model_ft, input_size = initialize_model('resnet152')
    feature_extractor = torch.nn.Sequential(*list(model_ft.children())[:-1]).to('cuda')

    transform = data_transform_color(input_size)
    trainset = datasets.CIFAR10(root, train=True, transform=transform, download=True)
    testset = datasets.CIFAR10(root, train=False, transform=transform, download=True)

    train_loader = DataLoader(trainset, batch_size=256,shuffle=False)
    test_loader = DataLoader(testset, batch_size=256,shuffle=False)

    train_features = []
    test_features = []
    train_targets = []
    test_targets = []

    feature_extractor.eval()
    with torch.no_grad():
        for data,target in train_loader:
            data = data.to('cuda')
            feature = feature_extractor(data)
            train_features.append(feature.cpu())
            train_targets.append(target.cpu())
        train_features = torch.cat(train_features,0).squeeze()
        train_targets = torch.cat(train_targets,0)
        for data,target in test_loader:
            data = data.to('cuda')
            feature = feature_extractor(data)
            test_features.append(feature.cpu())
            test_targets.append(target.cpu())

        test_features = torch.cat(test_features,0).squeeze()
        test_targets = torch.cat(test_targets,0)

    return [train_features,train_targets],[test_features,test_targets]


def extract_fmnist_feature(root):
    model_ft, input_size = initialize_model('resnet152')
    feature_extractor = torch.nn.Sequential(*list(model_ft.children())[:-1]).to('cuda')

    transform = data_transform_gray(input_size)
    trainset = datasets.FashionMNIST(root, train=True, transform=transform, download=True)
    testset = datasets.FashionMNIST(root, train=False, transform=transform, download=True)

    train_loader = DataLoader(trainset, batch_size=256,shuffle=False)
    test_loader = DataLoader(testset, batch_size=256,shuffle=False)

    train_features = []
    test_features = []
    train_targets = []
    test_targets = []

    feature_extractor.eval()
    with torch.no_grad():
        for data,target in train_loader:
            data = data.to('cuda')
            feature = feature_extractor(data)
            train_features.append(feature.cpu())
            train_targets.append(target.cpu())
        train_features = torch.cat(train_features,0).squeeze()
        train_targets = torch.cat(train_targets,0)
        for data,target in test_loader:
            data = data.to('cuda')
            feature = feature_extractor(data)
            test_features.append(feature.cpu())
            test_targets.append(target.cpu())

        test_features = torch.cat(test_features,0).squeeze()
        test_targets = torch.cat(test_targets,0)

    return [train_features,train_targets],[test_features,test_targets]

if __name__=='__main__':
    path = './DATA/'
    trainset, testset = extract_cifar_feature(path)
    torch.save(trainset,path+'/cifar10_feat/trainset_2048.pt')
    torch.save(testset, path+'/cifar10_feat/testset_2048.pt')
    trainset, testset = extract_fmnist_feature(path)
    torch.save(trainset,path+'/fmnist_feat/trainset_2048.pt')
    torch.save(testset, path+'/fmnist_feat/testset_2048.pt')
    