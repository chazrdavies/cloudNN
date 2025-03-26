import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
import rasterio

import random

from utils.imagepreprocessing import normalize_band




class SatelliteSegmentationDataset(Dataset):

    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.augmentations = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.RandomResizedCrop(size=(512, 512), scale=(0.8, 1.0)),  # Random crop
        ])

    def __len__(self):
        return len(self.image_filenames)
    

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        

        try:
            with rasterio.open(img_path) as src:
                blue = normalize_band(src.read(2))
                green = normalize_band(src.read(3))
                red = normalize_band(src.read(4))
                nir = normalize_band(src.read(5))

                image = np.stack([red, green, blue, nir], axis=-1).astype(np.float32)
                # min_vals = image.min(axis=(0, 1), keepdims=True)
                # max_vals = image.max(axis=(0, 1), keepdims=True)
                # image = (image - min_vals) / (max_vals - min_vals)

            with rasterio.open(mask_path) as src:
                label = src.read().astype(np.int8)

                # cloud_shadow = ((label == 0) | (label == 1)).astype(np.int8)
                cloud = ((label == 5)).astype(np.int8)
                water = ((label == 2) | (label == 6) | (label == 1)).astype(np.int8)
                land = ((label == 4) | (label == 0)).astype(np.int8)
                snow = (label == 3).astype(np.int8)

                mask = np.stack([cloud, water, land, snow], axis=0)  # Fix concatenation issue
                mask = mask.squeeze()

                # cloud = (label == 5).astype(np.uint8)
                # not_cloud = (label != 5).astype(np.uint8)

                # mask = np.stack([cloud, not_cloud])

                # mask = mask.squeeze()


                

            image = self.transform(image)
            mask = torch.tensor(mask, dtype=torch.long)

            if self.augmentations:
                seed = np.random.randint(0, 10000)
                torch.manual_seed(seed)
                image = self.augmentations(image)
                torch.manual_seed(seed)
                mask = self.augmentations(mask)
                mask = torch.argmax(mask, dim=0)


            return image, mask

        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            return None  # Return None to identify failing samples

        
def make_data_loader(image_dir, mask_dir, batch_size = 10, train_ratio = 0.8):
     

    dataset = SatelliteSegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir
    )


    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
    









