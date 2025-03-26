import torch
import torch.nn.functional as F
import torch.nn as nn

class DownSample(nn.Module):

    """
    Performs a convolution and a pool

    Returns Down and pool
    """

    def __init__(self, in_channels, out_channels):
        super.__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(inplace=True)
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride = 2)

    def forward(self, x):

        down = self.conv(x)
        pool = self.pool(down)

        return down, pool


class UpSample(nn.Module):

    def __init__(self, in_channels, out_channels):
        super.__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride = 1, padding = 'same'),
            nn.ReLU(inplace=True)
        )
        

    def forward(self, x1, x2):
        
        upsamp = self.up(x1)

        if x1.size() != x2.size():
            print("x1 and x2 are not the same size")
            raise Exception("x1 and x2 aren't the same size")

        concat= torch.cat([upsamp,x2], 1)
        
        return self.conv(concat)

    

        
class FCN(nn.Module):
    """
    FCN for dense prediction

    args: 
        in_channels (int)
        num_class (int)
        upsampled_predictions: 31, 16, or 8
    """

    def __init__(self, in_channels, num_classes, upsampled_predictions = "FCN_32"):
        super.__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.upsampled_predictions = upsampled_predictions

        self.down1 = DownSample(in_channels, out_channels = 32)
        self.down2 = DownSample(in_channels = 32, out_channels = 64)
        self.down3 = DownSample(in_channels = 64, out_channels = 128)
        self.down4 = DownSample(in_channels=128, out_channels=256)
        self.down5 = DownSample(in_channels= 256, out_channels=512)

        if self.upsampled_predictions == 'FCN_32': 
            self.decoder = nn.ConvTranspose2d(512, self.num_classes, kernel_size=32, stride=32, padding = 16)
        elif self.upsampled_predictions == 'FCN_16':
            self.decoder = nn.Sequential(
                UpSample(512, 256),
                nn.ConvTranspose2d(256, self.num_classes, kernel_size=16, stride=16, padding = 8)
            )

        elif self.upsampled_predictions == 'FCN_8':
            self.decoder = nn.Sequential(
                UpSample(512, 256),
                UpSample(256, 128),
                nn.ConvTranspose2d(128, self.num_classes, kernel_size=8, stride=8, padding = 4)
            )
            
        else:
            print("{self.upsampled_predictions} is not Valid")

        self.out = nn.Softmax(dim=1)

    def forward(self, x):


        down_1, pool_1 = self.down1(x)
        down_2, pool_2 = self.down1(pool_1)
        down_3, pool_3 = self.down1(pool_2)
        down_4, pool_4 = self.down1(pool_3)
        down_5, pool_5 = self.down1(pool_4)

        if self.upsampled_predictions == 'FCN_32': 
            final = self.decoder(pool_5)
        elif self.upsampled_predictions == 'FCN_16':
            self.decoder = nn.Sequential(
                UpSample(512, 256),
                nn.ConvTranspose2d(256, self.num_classes, kernel_size=16, stride=16, padding = 8)
            )
        else:
            self.decoder = nn.Sequential(
                UpSample(512, 256),
                UpSample(256, 128),
                nn.ConvTranspose2d(128, self.num_classes, kernel_size=8, stride=8, padding = 4)
            )
            


        return self.out(final)






        

        
