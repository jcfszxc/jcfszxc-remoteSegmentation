import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, Bottleneck


class CoordAtt(nn.Module):
    def __init__(self, in_channels):
        super(CoordAtt, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = in_channels // 2

        self.conv1 = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(self.inter_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(self.inter_channels, self.inter_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(self.inter_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(self.inter_channels, self.in_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(self.in_channels)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, _, h, w = x.size()
        # Spatial descriptor generation
        feats = self.conv1(x)
        feats = self.bn1(feats)
        feats = self.relu1(feats)

        feats = self.conv2(feats)
        feats = self.bn2(feats)
        feats = self.relu2(feats)

        feats = self.conv3(feats)
        feats = self.bn3(feats)

        feats = feats.view(b, self.in_channels, -1)
        feats = feats.permute(0, 2, 1)

        # Coordinate attention mechanism
        row_channel_wise = self.sigmoid(feats.max(dim=1)[0])
        col_channel_wise = self.sigmoid(feats.max(dim=2)[0])

        row_channel_wise = row_channel_wise.view(b, h, 1, -1)
        col_channel_wise = col_channel_wise.view(b, 1, w, -1)

        feats = torch.matmul(row_channel_wise, col_channel_wise)
        feats = feats.permute(0, 3, 1, 2)

        return x * feats


class ResNetWithCA(ResNet):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNetWithCA, self).__init__(block, layers, num_classes=num_classes,
                                           zero_init_residual=zero_init_residual)

        self.ca = CoordAtt(2048)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self._forward_impl(x)
        x = self.ca(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x
