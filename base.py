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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
  print("Using the GPU!")
else:
  print("WARNING: Could not find GPU! Using CPU only. If you want to enable GPU, please to go Edit > Notebook Settings > Hardware Accelerator and select GPU.")

# %%
# Data
# -----
LATENT_DIMS = 2
EPOCHS = 100
N_TRAIN = 500
N_VAL = 100

normal_train_loader, normal_val_loader, normal_train_labels = example_mnist_data(N_TRAIN, N_VAL)
encoder_1 = architectures.Encoder(latent_dims=LATENT_DIMS, feature_size=392)
encoder_2 = architectures.Encoder(latent_dims=LATENT_DIMS, feature_size=392)
decoder_1 = architectures.Decoder(latent_dims=LATENT_DIMS, feature_size=392)
decoder_2 = architectures.Decoder(latent_dims=LATENT_DIMS, feature_size=392)

# %%
# Simple CCA
# ----------------------------
latent_dims = 2

train_list = []
X,Y=[],[]
for i in normal_train_loader:
  for j in i['views'][0]:
    X.append(j.numpy())
  for j in i['views'][1]:
    Y.append(j.numpy())
X,Y = np.array(X), np.array(Y)

cca = CCA(latent_dims=latent_dims).fit((X, Y))
cca_result = cca.transform((X, Y))
pairplot_label(cca_result, normal_train_labels, title="Simple CCA")
plt.show()


# %%
# Deep CCA
# ----------------------------
dcca = DCCA(latent_dims=LATENT_DIMS, encoders=[encoder_1, encoder_2])
trainer = pl.Trainer(
    max_epochs=EPOCHS,
    enable_checkpointing=False,
    log_every_n_steps=1,
)

trainer.fit(dcca, normal_train_loader, normal_val_loader)
dcca_multiview_result = dcca.transform(normal_train_loader)
pairplot_label(dcca_multiview_result, normal_train_labels, title="DCCA")
plt.show()



# %%
# Deep CCAE
# ----------------------------
dccae = DCCAE(latent_dims=LATENT_DIMS, encoders=[encoder_1, encoder_2],decoders=[decoder_1,decoder_2])
trainer = pl.Trainer(
    max_epochs=EPOCHS,
    enable_checkpointing=False,
    log_every_n_steps=1,
)

trainer.fit(dccae, normal_train_loader, normal_val_loader)
dccae_multiview_result=dccae.transform(normal_train_loader)
pairplot_label(dccae_multiview_result, normal_train_labels, title="DCCAE")

plt.show()

# %%
# Deep CCA by Barlow Twins
# ----------------------------------------------
barlowtwins = BarlowTwins(latent_dims=LATENT_DIMS, encoders=[encoder_1, encoder_2])
trainer = pl.Trainer(
    max_epochs=EPOCHS,
    enable_checkpointing=False,
    log_every_n_steps=1,
)
trainer.fit(barlowtwins, normal_train_loader, normal_val_loader)
pairplot_label(
    barlowtwins.transform(normal_train_loader), normal_train_labels, title="DCCA by Barlow Twins"
)
plt.show()


# calculating error rate for MNIST

# training SVM
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
linear_svc = svm.LinearSVC()
linear_svc.fit(dcca_multiview_result[:][0][:], normal_train_labels)
#linear_svc.decision_function(dcca_multiview_result[:][0][:])

kfold=KFold(n_splits=10, shuffle=True, random_state=0)
# Simple CCA Score
simple_cca_svc = svm.LinearSVC()
simple_score = cross_val_score(linear_svc, cca_result[:][0][:], normal_train_labels , cv=kfold)
print(sum(simple_score)/10)

# DCCA score
linear_svc = svm.LinearSVC()
linear_scores = cross_val_score(linear_svc,dcca_multiview_result[:][0][:], normal_train_labels , cv=kfold)
print(sum(linear_scores)/10)

# DCCAE score
linear_svc_dccae = svm.LinearSVC()
linear_svc_dccae.fit(dccae_multiview_result[:][0][:], normal_train_labels)
dccae_score = cross_val_score(linear_svc,dccae_multiview_result[:][0][:], normal_train_labels , cv=kfold)
print(sum(dccae_score)/10)