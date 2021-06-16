import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ConvBnBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                padding=1,
                stride=stride,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ConvDwBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ConvDwBlock, self).__init__()

        self.block = nn.Sequential(
            # depth wise convolution signified by groups = in_channels
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(3, 3),
                stride=stride,
                padding=1,
                groups=in_channels,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class MobileNetV1(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(MobileNetV1, self).__init__()
        self.convBlock = nn.Sequential(
            self.createConvBnBlock(3, 32, 2),
            self.createConvDwBlock(32, 64, 1),
            self.createConvDwBlock(64, 128, 2),
            self.createConvDwBlock(128, 128, 1),
            self.createConvDwBlock(128, 256, 2),
            self.createConvDwBlock(256, 256, 1),
            self.createConvDwBlock(256, 512, 2),
            self.createConvDwBlock(512, 512, 1),
            self.createConvDwBlock(512, 512, 1),
            self.createConvDwBlock(512, 512, 1),
            self.createConvDwBlock(512, 512, 1),
            self.createConvDwBlock(512, 512, 1),
            self.createConvDwBlock(512, 1024, 2),
            self.createConvDwBlock(1024, 1024, 2),
        )

        # self.avgPool = nn.AvgPool2d(kernel_size=(7,7), stride=1)
        self.fc = nn.Linear(in_features=1024, out_features=num_classes)

    def forward(self, x):
        x = self.convBlock(x)
        # x = self.avgPool(x)
        x = F.avg_pool2d(x, 7)

        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

    def createConvBnBlock(self, in_channels, out_channels, stride):
        block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                padding=1,
                stride=stride,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        return block

    def createConvDwBlock(self, in_channels, out_channels, stride):
        block = nn.Sequential(
            # depth wise convolution signified by groups = in_channels
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(3, 3),
                stride=stride,
                padding=1,
                groups=in_channels,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        return block


if __name__ == "__main__":
    model = MobileNetV1()
    x = torch.randn(3, 3, 224, 224)
    print(model(x).shape)
