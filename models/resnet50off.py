from torch import nn
import torch.nn.functional as F
from torchvision import models


class Encoder(nn.Module):
    def __init__(self, in_channels=3, h=256, dropout=0.5):
        super(Encoder, self).__init__()
        
        resnetModel = models.resnet50(pretrained=True)
        feature_map = list(resnetModel.children())
        feature_map.pop()
        self.feature_extractor = nn.Sequential(*feature_map)

    def forward(self, x):
        x = x.expand(x.data.shape[0], 3, 227, 227)
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return x


class Classifier(nn.Module):
    def __init__(self, n_classes, dropout=0.5):
        super(Classifier, self).__init__()
        self.l1 = nn.Linear(2048, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = self.l1(x)
        return x


class CNN(nn.Module):
    def __init__(self, in_channels=3, n_classes=31, target=False):
        super(CNN, self).__init__()
        self.encoder = Encoder(in_channels=in_channels)
        self.classifier = Classifier(n_classes)
        if target:
            for param in self.classifier.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, h=500, args=None):
        super(Discriminator, self).__init__()
        self.l1 = nn.Linear(2048, h)
        self.l2 = nn.Linear(h, h)
        self.l3 = nn.Linear(h, 2)
        self.slope = args.slope

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = F.leaky_relu(self.l1(x), self.slope)
        x = F.leaky_relu(self.l2(x), self.slope)
        x = self.l3(x)
        return x
