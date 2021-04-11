import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class C1(nn.Module):
    def __init__(self):
        super(C1, self).__init__()
        self.c1 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "c1",
                        nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5)),
                    ),
                    ("relu1", nn.ReLU()),
                    ("maxpool1", nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
                ]
            )
        )

    def forward(self, img):
        out = self.c1(img)
        return out


class C2(nn.Module):
    def __init__(self):
        super(C2, self).__init__()
        self.c2 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "c2",
                        nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5)),
                    ),
                    ("relu2", nn.ReLU()),
                    ("maxpool2", nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
                ]
            )
        )

    def forward(self, img):
        out = self.c2(img)
        return out


class C3(nn.Module):
    def __init__(self):
        super(C3, self).__init__()
        self.c3 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "c3",
                        nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5, 5)),
                    ),
                    ("relu3", nn.ReLU()),
                ]
            )
        )

    def forward(self, img):
        out = self.c3(img)
        return out


class F4(nn.Module):
    def __init__(self):
        super(F4, self).__init__()
        self.f4 = nn.Sequential(
            OrderedDict(
                [
                    ("f4", nn.Linear(in_features=120, out_features=84)),
                    ("relu4", nn.ReLU()),
                ]
            )
        )

    def forward(self, img):
        out = self.f4(img)
        return out


class F5(nn.Module):
    def __init__(self):
        self.f5 = nn.Sequential(
            OrderedDict(
                [
                    ("f5", nn.Linear(in_features=84, out_features=10)),
                    ("softmax5", nn.LogSoftmax(dim=-1)),
                ]
            )
        )

    def forward(self, img):
        out = self.f5(img)
        return out


class LeNet(nn.Module):
    def __init__(self):

        self.c1 = C1()

        self.c2_1 = C2()
        self.c2_2 = C2()

        self.c3 = C3()

        self.f4 = F4()
        self.f5 = F5()

    def forward(self, img):
        out = self.c1(img)

        x = self.c2_1(out)
        out = self.c2_2(out)

        out += x

        out = self.c3(out)
        out = out.view(img.size(0), -1)

        out = self.f4(out)
        out = self.f5(out)

        return out
