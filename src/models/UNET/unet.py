import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Notes

Add dropout layer?
"""


class DoubleConv(nn.Module):
    """Applies a double convolution to data

    Two convolution layers with (3x3) kernel and two Relu Activations

    Attributes:
        in_channel: # of channels coming into layer
        out_channel: # of channels going out
    """

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        


    def forward(self, x):
        return self.conv(x)



class DownSample(nn.Module):
    """Applies a Down Sample to data

    One double convolution then a max pool with a 2x2 kernel and stride of 2

    Attributes:
        in_channel: # of channels coming into layer
        out_channel: # of channels going out
    """

    def __init__(self, in_channel, out_channel):
        super(DownSample, self).__init__()

        self.double_conv = DoubleConv(in_channel, out_channel)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.double_conv(x)  # This is a tensor, not a tuple
        pool = self.max_pool(down)  # Now applying max pooling on the tensor


        return down, pool

        
class UpSample(nn.Module):
    """Applies a Down Sample to data

    One double convolution then a max pool with a 2x2 kernel and stride of 2

    Attributes:
        in_channel: # of channels coming into layer
        out_channel: # of channels going out
    """

    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channel, in_channel//2, kernel_size=2, stride=2)
        # maybe add dropout
        self.conv = DoubleConv(in_channel, out_channel)

    def forward(self, x1, x2):

        x1 = self.up(x1)


        if x1.size() != x2.size():
            x2 = F.interpolate(x2, size=x1.size()[2:], mode='bilinear', align_corners=False)


        x2 = torch.cat([x1, x2], 1)

        return self.conv(x2)


class MiniUnet(nn.Module):
    """
    A CNN model for image classification.

    Args:
        in_channels (int): The number of input channels.
        num_classes (int): The number of output classes.

    Attributes:

        
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes



        self.down1 = DownSample(self.in_channels, 32)
        self.down2 = DownSample(32, 64)
        self.down3 = DownSample(64, 128)

        self.bottle_neck = DoubleConv(128, 256)
        ## should i add a pooling after bottleneck?

        self.up1 = UpSample(256, 128)
        self.up2 = UpSample(128, 64)
        self.up3 = UpSample(64, 32)


        self.final_layer = nn.Conv2d(32, num_classes, kernel_size=1)

        # self.out = nn.Softmax(dim=1) # softmax accross class dimension




    def forward(self, x):
        """The forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, num_classes).
        """
        # Apply the convolutional layers and the pooling layer

        down_1, pool_1 = self.down1(x)
        down_2, pool_2 = self.down2(pool_1)
        down_3, pool_3 = self.down3(pool_2)

        b = self.bottle_neck(pool_3)

        up_1 = self.up1(b, down_3)
        up_2 = self.up2(up_1, down_2)
        up_3 = self.up3(up_2, down_1)


        logits = self.final_layer(up_3)

        return logits




