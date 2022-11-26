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
from utils import Encoder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
  print("Using the GPU!")
else:
  print("WARNING: Could not find GPU! Using CPU only. If you want to enable GPU, please to go Edit > Notebook Settings > Hardware Accelerator and select GPU.")

def DCCA():
    # ----------------------------
    encoder_1 = Encoder(latent_dims=LATENT_DIMS)
    encoder_2 = Encoder(latent_dims=LATENT_DIMS)
    decoder_1 = architectures.Decoder(latent_dims=LATENT_DIMS, feature_size=392)
    decoder_2 = architectures.Decoder(latent_dims=LATENT_DIMS, feature_size=392)

    dcca = DCCA(latent_dims=LATENT_DIMS, encoders=[encoder_1, encoder_2])
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        enable_checkpointing=False,
        log_every_n_steps=1,
    )

    trainer.fit(dcca, train_loader, val_loader)
    dcca_multiview_result = dcca.transform(train_loader)
    pairplot_label(dcca_multiview_result, train_labels, title="DCCA")
    plt.show()

    # put cluster result with label as a dataset

    # CNN/MLP to train the dataset
