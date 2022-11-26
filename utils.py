# !pip install torchsummary
# !pip install cca-zoo[deep]
# !pip install multiviewdata
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import itertools
from matplotlib import image
import glob as glob
from PIL import Image
import cca_zoo

from torchvision.datasets import MNIST

from multiviewdata.torchdatasets import SplitMNIST, NoisyMNIST
from torch.utils.data import Subset
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary

from multiviewdata.torchdatasets import SplitMNIST, NoisyMNIST
from torch.utils.data import Subset
import numpy as np

import pytorch_lightning as pl
from matplotlib import pyplot as plt
from cca_zoo import CCA

from cca_zoo.deepmodels import (
    DCCA,
    DCCA_NOI,
    DCCA_SDL,
    DCCAE,
    BarlowTwins,
)
from cca_zoo.deepmodels import architectures
from cca_zoo.plotting import pairplot_label

from cca_zoo.data.deep import get_dataloaders

from torchvision import datasets, transforms
import torch

def get_dataloaders(
        dataset,
        val_dataset=None,
        batch_size=None,
        val_batch_size=None,
        drop_last=True,
        val_drop_last=False,
        shuffle_train=False,
        pin_memory=True,
        num_workers=0,
        persistent_workers=True,
    ):
    """
    A utility function to allow users to quickly get hold of the dataloaders required by pytorch lightning
    :param dataset: A CCA dataset used for training
    :param val_dataset: An optional CCA dataset used for validation
    :param batch_size: batch size of train loader
    :param val_batch_size: batch size of val loader
    :param num_workers: number of workers used
    :param pin_memory: pin memory used by pytorch - True tends to speed up training
    :param shuffle_train: whether to shuffle training data
    :param val_drop_last: whether to drop the last incomplete batch from the validation data
    :param drop_last: whether to drop the last incomplete batch from the train data
    :param persistent_workers: whether to keep workers alive after dataloader is destroyed
    """
    if num_workers == 0:
        persistent_workers = False
    if batch_size is None:
        batch_size = len(dataset)
    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle_train,
        persistent_workers=persistent_workers,
    )
    if val_dataset:
        if val_batch_size is None:
            val_batch_size = len(val_dataset)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            drop_last=val_drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        return train_dataloader, val_dataloader
    return train_dataloader


def example_mnist_data(n_train, n_val, batch_size=50, val_batch_size=10, type="split"):
    if type == "split":
        train_dataset = SplitMNIST(
            root="", mnist_type="MNIST", train=True, download=True
        )
    else:
        train_dataset = NoisyMNIST(
            root="", mnist_type="MNIST", train=True, download=True
        )
    # Adding noise to the mnist
    for i in train_dataset:
      noise = np.random.normal(0.0, 0.1, i['views'][1].shape)
      noise = torch.from_numpy(noise)
      noisy_view=list(i['views'])
      noisy_view[1]+=noise
      i['views']=noisy_view

    val_dataset = Subset(train_dataset, np.arange(n_train, n_train + n_val))
    train_dataset = Subset(train_dataset, np.arange(n_train))
    train_loader, val_loader = get_dataloaders(
        train_dataset, val_dataset, batch_size=batch_size, val_batch_size=val_batch_size
    )
    train_labels = train_loader.collate_fn(
        [train_dataset.dataset[idx]["label"] for idx in train_dataset.indices]
    ).numpy()
    return train_loader, val_loader, train_labels



# Plotting Module
import seaborn as sns
import pandas as pd

def plotting(view_1, view_2, labels):
  # view 1
  view_dict = {"x_coor_view1":[], "y_coor_view1":[],"x_coor_view2":[], "y_coor_view2":[], "labels":[]}
  for point1, point2, label in zip(view_1, view_2, labels):
    view_dict["x_coor_view1"].append(point1[0])
    view_dict["y_coor_view1"].append(point1[1])
    view_dict["x_coor_view2"].append(point2[0])
    view_dict["y_coor_view2"].append(point2[1])
    view_dict["labels"].append(str(label))
  view_df = pd.DataFrame(view_dict)
  sns.set(rc={'figure.figsize':(36,18)})

  #subplot
  f, axes = plt.subplots(1, 2)
  axes[0].set_xlim(-2.0, 1.5)
  axes[0].set_ylim(-2.0, 1.5)
  axes[1].set_xlim(-2.0, 1.5)
  axes[1].set_ylim(-2.0, 1.5)
  sns.scatterplot(data=view_df, x="x_coor_view1", y="y_coor_view1", hue="labels", legend="full", ax=axes[0])
  sns.scatterplot(data=view_df, x="x_coor_view2", y="y_coor_view2", hue="labels", legend="full", ax=axes[1])

# view_00, view_01 = cca_result[0], cca_result[1]
# view_11, view_12 = dcca_multiview_result[0], dcca_multiview_result[1]
# view_21, view_22 = dccae_multiview_result[0], dccae_multiview_result[1]
# plotting(view_00, view_01, normal_train_labels)
# plotting(view_11, view_12, normal_train_labels)
# plotting(view_21, view_22, normal_train_labels)

# Dataloader


def load_haze_data(data_folder, batch_size, train, num_workers=0, **kwargs):
    transform = {
        'train': transforms.Compose(
            [transforms.Resize([256, 256]),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])]),
        'test': transforms.Compose(
            [transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])
    }
    data = datasets.ImageFolder(root=data_folder, transform=transform['train' if train else 'test'])
    data_loader = get_haze_data_loader(data, batch_size=batch_size, 
                                shuffle=True if train else False, 
                                num_workers=num_workers, **kwargs, drop_last=True if train else False)
    n_class = len(data.classes)

    return data_loader, n_class

def get_haze_data_loader(dataset, batch_size, shuffle=True, drop_last=False, num_workers=0, infinite_data_loader=False, **kwargs):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last, num_workers=num_workers, **kwargs)


# ENCODER SECTION

from abc import abstractmethod

class _BaseEncoder(torch.nn.Module):
    @abstractmethod
    def __init__(self, latent_dims: int, variational: bool = False):
        super(_BaseEncoder, self).__init__()
        self.variational = variational
        self.latent_dims = latent_dims

    @abstractmethod
    def forward(self, x):
        pass


class Encoder(_BaseEncoder):
    def __init__(
            self,
            latent_dims: int,
            variational: bool = False,
            feature_size: int = 1024,
            layer_sizes: tuple = None,
            activation=nn.LeakyReLU(),
            dropout=0,
    ):
        super(Encoder, self).__init__(latent_dims, variational=variational)
        if layer_sizes is None:
            layer_sizes = (128,)
        layer_sizes = (feature_size,) + layer_sizes + (latent_dims,)
        layers = []
        # other layers
        self.conv1 = torch.nn.Conv2d(1, 32, 5)  # chnl-in, out, krnl
        self.conv2 = torch.nn.Conv2d(32, 64, 5)
        self.pool1 = torch.nn.MaxPool2d(2, stride=2)
        self.pool2 = torch.nn.MaxPool2d(2, stride=2)
        for l_id in range(len(layer_sizes) - 2):
            layers.append(
                torch.nn.Sequential(
                    nn.Dropout(p=dropout),
                    torch.nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                    activation,
                )
            )
        self.layers = torch.nn.Sequential(*layers)

        if self.variational:
            self.fc_mu = torch.nn.Sequential(
                nn.Dropout(p=dropout), torch.nn.Linear(layer_sizes[-2], layer_sizes[-1])
            )
            self.fc_var = torch.nn.Sequential(
                nn.Dropout(p=dropout), torch.nn.Linear(layer_sizes[-2], layer_sizes[-1])
            )
        else:
            self.fc = torch.nn.Sequential(
                nn.Dropout(p=dropout), torch.nn.Linear(layer_sizes[-2], layer_sizes[-1])
            )

    def forward(self, x):
      # x:[bs,1,28,28]
      # convolution phase         # x is [bs, 1, 28, 28]
        z = torch.relu(self.conv1(x))   # Size([bs, 32, 24, 24])
        z = self.pool1(z)           # Size([bs, 32, 12, 12])
        z = torch.relu(self.conv2(z))   # Size([bs, 64, 8, 8])
        z = self.pool2(z)           # Size([bs, 64, 4, 4])
   
        # neural network phase
        z = z.reshape(-1, 1024)     # Size([bs, 1024])
        x = self.layers(z)
        if self.variational:
            mu = self.fc_mu(x)
            logvar = self.fc_var(x)
            return mu, logvar
        else:
            x = self.fc(x)
            return x

def load_mnist_data(n_train, batch_size=50, val_batch_size=10, noise_std=0.1):
    # Noisy Multiview MNIST Dataset, Len=60000

    transform = transforms.Compose(
            [transforms.Resize([28, 28]),
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                #                  std=[0.229, 0.224, 0.225])
                ])
    mnist_image = MNIST(root="",transform=transform,download=True)
    N=len(mnist_image)
    mnist_loader= get_dataloaders(
            mnist_image, batch_size=64,
        )

    our_dataset=[]
    labels=[]
    def add_noise(v):
        noise=np.random.normal(0, noise_std,  size=v.shape)
        v+=torch.from_numpy(noise)
        return torch.clamp(v, min=0, max=1.)

    for index,i in enumerate(mnist_image):
        data={'views':None,'label':None,'index':None}
        data['label']=i[1]
        data['index']=index
        data['views']=(i[0],add_noise(i[0]))
        our_dataset.append(data)
        labels.append(i[1])

    # our_mnist_loader=get_dataloaders(
    #         our_dataset, batch_size=10,
    #     )

    val_dataset = our_dataset[n_train:]
    train_dataset = our_dataset[:n_train]
    train_loader, val_loader = get_dataloaders(
            train_dataset, val_dataset, batch_size, val_batch_size
        )

    train_labels =np.array(labels[:(n_train//batch_size)*batch_size])

    # TODO: Add val_labels
    val_labels=np.array(labels[(n_train//batch_size)*batch_size:])
    
    return train_loader, val_loader, train_labels, val_labels