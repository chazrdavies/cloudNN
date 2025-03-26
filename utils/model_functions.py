
import src.models.UNET.unet as unet
import torch


# for loading in saved models

def load_mini_unet_model(path):

    model = unet.MiniUnet()
    model.load_state_dict(torch.load('path'))

    return model