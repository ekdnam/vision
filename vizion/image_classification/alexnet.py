import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class C1(nn.Module):
    def __init__(self):
        self.c1 = nn.Sequential(OrderedDict([
            ("c1", nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4)),
            ('relu1', nn.ReLU()),
            ('responseNormalized1', nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)),
            ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2))
        ]))

    def forward(self, x):
        out = self.c1(x)
        return out


class C2(nn.Module):
    def __init__(self):
        self.c2 = nn.Sequential(OrderedDict([
            ('c2', nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2, groups=2)),
            # ('c2', nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2, stride=2, groups=2)),
            ('relu2', nn.ReLU()),
            ('responseNormalized2', nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)),
            ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2))
        ]))

    def forward(self, x):
        out = self.c2(x)
        return out

class C3(nn.Module):
    def __init__(self):
        self.c3 = nn.Sequential(OrderedDict([
            ("c3", nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)),
            ('relu3', nn.ReLU())
        ]))

    def forward(self, x):
        out = self.c3(x)
        return out

class C4(nn.Module):
    def __init__(self):
        self.c4 = nn.Sequential(OrderedDict([
            ("c4", nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1, groups=2)),
            ('relu4', nn.ReLU())
        ]))

    def forward(self, x):
        out = self.c4(x)
        return out

class C5(nn.Module):
    def __init__(self):
        self.c5 = nn.Sequential(OrderedDict([
            ("c5", nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1, groups=2)),
            ('relu5', nn.ReLU()),
            ('maxpool5', nn.MaxPool2d(kernel_size=3, stride=2))
        ]))

    def forward(self, x):
        out = self.c5(x)
        return out
        
class F6(nn.Module):
    def __init__(self):
        self.f6 = nn.Sequential(OrderedDict([
            ("linear6", nn.Linear(256*6*6, 4096)),
            ('relu6', nn.ReLU()),
            ('dropout6', nn.Dropout())
        ]))

    def forward(self, x):
        out = self.f6(x)
        return out

class F7(nn.Module):
    def __init__(self):
        self.f7 = nn.Sequential(OrderedDict([
            ("linear7", nn.Linear(4096, 4096)),
            ('relu7', nn.ReLU()),
            ('dropout7', nn.Dropout())
        ]))

    def forward(self, x):
        out = self.f7(x)
        return out

class F8(nn.Module):
    def __init__(self):
        self.f8 = nn.Sequential(OrderedDict([
            ("linear6", nn.Linear(4096, 1000))
        ]))

    def forward(self, x):
        out = self.f7(x)
        return out
