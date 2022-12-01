
from torchvision.datasets import CIFAR10
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import torchvision.datasets as datasets
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def get_cifar10():
    path='c:\\Users\\cs843\\Documents\\PhD\\November\\App\\Datasets\\Cifar10'
    # Prepare CIFAR-10 dataset
    trainset = CIFAR10(path,train=True, download=True, transform=transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    testset = CIFAR10(path,train=False, download=True, transform=transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    X_train, y_train = my_fun(trainset)
    X_test, y_test =my_fun(testset)
    return X_train, X_test, y_train, y_test

def my_fun(dataset):
    X=[]
    y=[]
    for feature, label in iter(dataset):
        X.append(feature)
        y.append(label)
    return torch.stack(X).numpy(), y

def get_binary_data(name='make_moons', test_size=0.3):
    #test plotable data 
    if name=='make_moons':
        X, y = make_moons(noise=0.3, random_state=0)
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    elif name=='make_circles':
        X, y = make_circles(noise=0.3, random_state=0)
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    elif name=="make_classification":
        X, y = make_classification(noise=0.3, random_state=0)
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    return X,y, X_train, X_test, y_train, y_test