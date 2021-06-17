import torch
import torch.nn as nn
import torchvision
from collections import OrderedDict


class Fire(nn.Module):
    def __init__(
        self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels
    ):
        super(Fire, self).__init__()
        self.in_channels = in_channels
        self.squeeze_channels = squeeze_channels
        self.expand1x1_channels = expand1x1_channels
        self.expand3x3_channels = expand3x3_channels

        self.squeeze_module = nn.Sequential(
            OrderedDict(
                [
                    (
                        "squeeze",
                        nn.Conv2d(in_channels, squeeze_channels, kernel_size=(1, 1)),
                    ),
                    ("squeeze_activation", nn.ReLU(inplace=True)),
                ]
            )
        )

        self.expand1x1_module = nn.Sequential(
            OrderedDict(
                [
                    (
                        "expand1x1_conv",
                        nn.Conv2d(
                            squeeze_channels, expand1x1_channels, kernel_size=(1, 1)
                        ),
                    ),
                    ("expand1x1_activation", nn.ReLU(inplace=True)),
                ]
            )
        )

        self.expand3x3_module = nn.Sequential(
            OrderedDict(
                [
                    (
                        "expand3x3_conv",
                        nn.Conv2d(
                            squeeze_channels, expand3x3_channels, kernel_size=(1, 1)
                        ),
                    ),
                    ("expand3x3_activation", nn.ReLU(inplace=True)),
                ]
            )
        )

    def forward(self, x):
        x = self.squeeze_module(x)
        return torch.cat([self.expand1x1_module(x), self.expand3x3_module(x)], 1)


class SqueezeNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=7, stride=2)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fire1 = Fire(
            in_channels=96,
            squeeze_channels=16,
            expand1x1_channels=64,
            expand3x3_channels=64,
        )
        self.fire2 = Fire(
            in_channels=128,
            squeeze_channels=16,
            expand1x1_channels=64,
            expand3x3_channels=64,
        )
        self.fire3 = Fire(
            in_channels=128,
            squeeze_channels=32,
            expand1x1_channels=128,
            expand3x3_channels=128,
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fire4 = Fire(
            in_channels=256,
            squeeze_channels=32,
            expand1x1_channels=128,
            expand3x3_channels=128,
        )
        self.fire5 = Fire(
            in_channels=256,
            squeeze_channels=48,
            expand1x1_channels=192,
            expand3x3_channels=192,
        )
        self.fire6 = Fire(
            in_channels=384,
            squeeze_channels=48,
            expand1x1_channels=192,
            expand3x3_channels=192,
        )
        self.fire7 = Fire(
            in_channels=384,
            squeeze_channels=64,
            expand1x1_channels=256,
            expand3x3_channels=256,
        )
        self.maxpool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        self.fire8 = Fire(
            in_channels=512,
            squeeze_channels=64,
            expand1x1_channels=256,
            expand3x3_channels=256,
        )
        self.dropout1 = nn.Dropout(p=0.5)
        self.conv2 = nn.Conv2d(
            in_channels=512, out_channels=num_classes, kernel_size=(1, 1)
        )
        self.relu2 = nn.ReLU()
        self.avgpool1 = nn.AvgPool2d(kernel_size=12, stride=1)

    def forward(self, x):
        # print("Input " + str(x.size()))
        x = self.conv1(x)
        # print("Conv1 " + str(x.size()))
        x = self.relu1(x)
        # print("ReLU " + str(x.size()))
        x = self.maxpool1(x)
        # print("Maxpool1 " + str(x.size()))
        x = self.fire1(x)
        # print("Fire1 " + str(x.size()))
        x = self.fire2(x)
        # print("Fire2 " + str(x.size()))
        x = self.fire3(x)
        # print("Fire3 " + str(x.size()))
        x = self.maxpool2(x)
        # print("Maxpool2 " + str(x.size()))
        x = self.fire4(x)
        # print("Fire4 " + str(x.size()))
        x = self.fire5(x)
        # print("Fire5 " + str(x.size()))
        x = self.fire6(x)
        # print("Fire6 " + str(x.size()))
        x = self.fire7(x)
        # print("Fire7 " + str(x.size()))
        x = self.maxpool3(x)
        # print("Maxpool3 " + str(x.size()))
        x = self.fire8(x)
        # print("Fire8 " + str(x.size()))
        x = self.dropout1(x)
        # print("Dropout1 " + str(x.size()))
        x = self.conv2(x)
        # print("Conv2 " + str(x.size()))
        x = self.relu2(x)
        # print("Relu2 " + str(x.size()))
        x = self.avgpool1(x)
        # print("Avgpool1 " + str(x.size()))

        # print("Final " + str(x.size()))
        return x.view(x.size(0), self.num_classes)


if __name__ == "__main__":
    model = SqueezeNet().to("cpu")
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.size())
