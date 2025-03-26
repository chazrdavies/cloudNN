import torch
import numpy as np

import src.SatelliteSegmentationDataset as dataset
import src.models.UNET.unet as unet

from datetime import datetime
import sys

import utils.config as config
from utils.model_functions import load_mini_unet_model

def test_model():
    pass


import torchvision
import torch
import numpy as np
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

import src.SatelliteSegmentationDataset as dataset
import src.models.UNET.unet as unet

from datetime import datetime
#import datetime
import sys

import utils.config as config
import matplotlib.pyplot as plt


# for loading in saved models

def load_mini_unet_model(path):

    model = unet.MiniUnet()
    model.load_state_dict(torch.load('path'))

    return model
        

model_path = "results/mini_unet_v1.pth"

model = load_mini_unet_model(model_path)


train_loader, val_loader = dataset.make_data_loader(config.IMAGE_DIR, config.MASK_DIR, batch_size=config.BATCH_SIZE, train_ratio=config.TRAIN_SPLIT)

batch = next(iter(val_loader))

inputs, labels = batch

outputs = model(inputs)

output = outputs[0]

prediction = torch.argmax(output, dim=0)

plt.imshow(prediction, cmap='gray')
plt.title("Prediction")
plt.show()

mask = torch.argmax(labels[0], dim=0)
plt.imshow(mask, cmap='gray')
plt.title("Label")
plt.show()