import numpy as np



def compute_indices(image):
    nir = image[3, :, :].astype(np.float32)  # NIR Channel
    red = image[0, :, :].astype(np.float32)  # Red Channel
    green = image[1, :, :].astype(np.float32)  # Green Channel

    ndvi = (nir - red) / (nir + red + 1e-6)  # Avoid division by zero
    ndwi = (green - nir) / (green + nir + 1e-6)
    
    return ndvi, ndwi



