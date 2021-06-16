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

        # self.model = nn.Sequential(OrderedDict([
        #     ('conv1', nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(7,7), stride=2)),
        #     ('relu1', nn.ReLU(inplace=True)),
        #     ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2)),
        #     ('fire1',Fire(in_channels=96, squeeze_channels=16, expand1x1_channels=64, expand3x3_channels=64)),
        #     ('fire2', Fire(in_channels=128, squeeze_channels=16, expand1x1_channels=64, expand3x3_channels=64)),
        #     ('fire3', Fire(in_channels=128, squeeze_channels=32, expand1x1_channels=128, expand3x3_channels=128)),
        #     ('maxpool2', nn.MaxPool2d(kernel_size=(3,3), stride=2)),
        #     ('fire4', Fire(in_channels=256, squeeze_channels=32, expand1x1_channels=128, expand3x3_channels=128)),
        #     ('fire5', Fire(in_channels=256, squeeze_channels=48, expand1x1_channels=192, expand3x3_channels=192)),
        #     ('fire6', Fire(in_channels=384, squeeze_channels=48, expand1x1_channels=192, expand3x3_channels=192)),
        #     ('fire7',Fire(in_channels=384, squeeze_channels=64, expand1x1_channels=192, expand3x3_channels=192)),
        #     ('maxpool3', nn.MaxPool2d(kernel_size=(3,3), stride=2)),
        #     ('fire8', Fire(in_channels=512, squeeze_channels=64, expand1x1_channels=256, expand3x3_channels=256)),
        #     ('dropout1', nn.Dropout(p=0.5)),
        #     ('conv2', nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=(1,1))),
        #     ('relu2', nn.ReLU(inplace=True)),
        #     ('avgpool1',nn.AvgPool2d(kernel_size=13))
        # ]))
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=96, kernel_size=7, stride=2
        )
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
        # self.avgpool1 = nn.AvgPool2d(kernel_size=4, stride=4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.fire1(x)
        x = self.fire2(x)
        x = self.fire3(x)
        x = self.maxpool2(x)
        x = self.fire4(x)
        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.maxpool3(x)
        x = self.fire8(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        # x = self.avgpool1(x)

        return x.view(x.size(0), self.num_classes)

if __name__ == "__main__":
    model = SqueezeNet().to("cpu")
    x = torch.randn(3, 3, 224, 224)
    y = model(x)
    print(y.size())
