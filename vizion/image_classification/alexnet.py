import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class C1(nn.Module):
    def __init__(self):
        super(C1, self).__init__()
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
        super(C2, self).__init__()
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
        super(C3, self).__init__()
        self.c3 = nn.Sequential(OrderedDict([
            ("c3", nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)),
            ('relu3', nn.ReLU())
        ]))

    def forward(self, x):
        out = self.c3(x)
        return out

class C4(nn.Module):
    def __init__(self):
        super(C4, self).__init__()
        self.c4 = nn.Sequential(OrderedDict([
            ("c4", nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1, groups=2)),
            ('relu4', nn.ReLU())
        ]))

    def forward(self, x):
        out = self.c4(x)
        return out

class C5(nn.Module):
    def __init__(self):
        super(C5, self).__init__()
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
        super(F6, self).__init__()
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
        super(F7, self).__init__()
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
        super(F8, self).__init__()
        self.f8 = nn.Sequential(OrderedDict([
            ("linear6", nn.Linear(4096, 1000))
        ]))

    def forward(self, x):
        out = self.f7(x)
        return out

class Alexnet(nn.Module):
    def __init__(self):
        super(Alexnet, self).__init__()
        self.c1 = C1()
        self.c2 = C2()
        self.c3 = C3()
        self.c4 = C4()
        self.c5 = C5()
        self.f6 = F6()
        self.f7 = F7()
        self.f8 = F8()

    def forward(self, img):
        out = self.c1(img)
        out = self.c2(out)
        out = self.c3(out)
        out = self.c4(out)
        out = self.c5(out)

        out = out.view(img.size(0), -1)

        out = self.f6(out)
        out = self.f7(out)
        out = self.f8(out)

        return out

    
