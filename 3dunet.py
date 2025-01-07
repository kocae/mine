import torch
import torch.nn as nn
import torch.nn.functional as F

class myUNet(nn.Module):
    def __init__(self, in_channel=1, out_channel=2, training=True):
        super(myUNet, self).__init__()
        self.training = training
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder1 = nn.Sequential(
            nn.Conv3d(in_channel, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16), 
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, padding=1), # padding有必要吗？
            nn.BatchNorm3d(32),
            nn.ReLU()
        ) # 1->32
        self.encoder2 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32), 
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1), 
            nn.BatchNorm3d(64),
            nn.ReLU()
        ) # 32->62
        self.encoder3 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64), 
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, padding=1), 
            nn.BatchNorm3d(128),
            nn.ReLU()
        ) # 64->128
        self.encoder4 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128), 
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=3, padding=1), 
            nn.BatchNorm3d(256),
            nn.ReLU()
        ) # 128->256

        # decoder部分，每层分为拼接前的反卷积，拼接，拼接后的卷积三部分哈
        self.upconv3 = nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2)
        self.decoder3 = nn.Sequential(
            nn.Conv3d(128+256, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU()
        ) 

        self.upconv2 = nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv3d(128+64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )

        self.upconv1 = nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv3d(64+32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            # 最后输出
            nn.Conv3d(32, out_channels=2, kernel_size=1)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d((16,32,32)),  
            nn.Flatten(),
            nn.Linear(2*16*32*32, out_channel),
        )

    def forward(self, x):
        # encoder
        enc1 = self.encoder1(x) # channel=32
        pool1 = self.maxpool(enc1)

        enc2 = self.encoder2(pool1)
        pool2 = self.maxpool(enc2)

        enc3 = self.encoder3(pool2)
        pool3 = self.maxpool(enc3)

        enc4 = self.encoder4(pool3)


        # decoder
        up3 = self.upconv3(enc4)
        dec3 = torch.cat([enc3, up3], dim=1)
        dec3 = self.decoder3(dec3)
        
        up2 = self.upconv2(dec3)
        dec2 = torch.cat([enc2, up2], dim=1)
        dec2 = self.decoder2(dec2)

        up1 = self.upconv1(dec2)
        dec1 = torch.cat([enc1, up1], dim=1)
        dec1 = self.decoder1(dec1)

        out = self.classifier(dec1)
        print(f"max and min of out: {out.max()}, {out.min()}")

        return out
