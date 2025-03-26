import matplotlib.pyplot as plt
import torch
import utils.config as config
from src.SatelliteSegmentationDataset import make_data_loader

# Set paths
image_dir = config.IMAGE_DIR
mask_dir = config.MASK_DIR


def show_sample(image, mask):
    fig, axes = plt.subplots(1, 6, figsize=(15, 5))
    
    # Show each channel separately
    axes[0].imshow(image[0], cmap='Reds')  # Red
    axes[0].set_title("Red Channel")
    axes[1].imshow(image[1], cmap='Greens')  # Green
    axes[1].set_title("Green Channel")
    axes[2].imshow(image[2], cmap='Blues')  # Blue
    axes[2].set_title("Blue Channel")
    axes[3].imshow(image[3], cmap='Purples')  # NIR
    axes[3].set_title("NIR Channel")



    # Overlay mask (sum of all classes for visualization)
    mask_overlay = mask.argmax(axis=0)  
    axes[4].imshow(mask_overlay, cmap='gray')
    axes[4].set_title("Segmentation Mask")

    rgb_image = image[:3]  # This takes the first 3 channels (Red, Green, Blue)

    # Reorder the channels to match the expected shape (Height, Width, Channels)
    rgb_image = rgb_image.transpose(1, 2, 0)  # Transpose to (512, 512, 3)

    
    axes[5].imshow(rgb_image)
    axes[5].set_title("RGB Image")


    plt.show()


def check_label(label):
    print("Label Shape :" ,  label.shape)
    # Overlay mask (sum of all classes for visualization)
    plt.imshow(label, cmap='gray')
    plt.title("Segmentation Mask")
    plt.show()

    return
    



def check_image(image):
    print(f"Image shape {image.shape}")
    print("Max: ", torch.max(image[0]))
    print("Min: ", torch.min(image[0]))
    
    # rgb_nir = image[0]

    rgb_image = image[:3]  # This takes the first 3 channels (Red, Green, Blue)

    # Reorder the channels to match the expected shape (Height, Width, Channels)
    rgb_image = torch.permute(rgb_image, (1, 2, 0))

    plt.imshow(rgb_image)
    plt.title("RGB Image")
    plt.show()

    return


train_loader, test_loader = make_data_loader(image_dir, mask_dir, batch_size=10)

# Fetch one batch
# for images, masks in train_loader:
#     print(f"Image batch shape: {images.shape}")  # Should be (batch_size, 4, H, W)
#     print(f"Mask batch shape: {masks.shape}")    # Should be (batch_size, 4, H, W)
#     break

batch = next(iter(train_loader))  # Get a batch
inputs, labels = batch

for i in range(3):
    check_label(labels[i])
    check_image(inputs[i])