import torch
import torch.nn as nn
import torchvision
import numpy as np
from collections import OrderedDict

# modelyolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

class BaselineCNN(nn.Module):
    def __init__(self, img_dim=640, backbone=None,num_classes=26, num_kernels=3, input_filter=3, num_filters1=128, num_filters2=64, num_hidden=512, num_hidden2=256, pooling_dim=2, stride=1, padding=1, stridepool=2, paddingpool=0):
        super().__init__()
        flattendim = int((img_dim/stridepool) ** 2) * num_filters2
        
        self.backbone = None
        self.conv1 = nn.Conv2d(input_filter, num_filters1, kernel_size=num_kernels, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(num_filters1, num_filters2, kernel_size=num_kernels, stride=stride, padding=padding)
        self.maxpool = nn.MaxPool2d(kernel_size=pooling_dim, stride=stridepool, padding=paddingpool)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flattendim, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden2)
        self.classifier = nn.Linear(num_hidden2,num_classes)
        
        if(backbone):
            self.backbone = nn.Sequential(*backbone)
            
    def forward(self, x):
        self.out = self.conv1(x)
        self.out = self.conv2(self.out)
        self.out = self.maxpool(self.out)
        self.out = self.flatten(self.out)
        self.out = self.fc1(self.out)
        self.out = self.fc2(self.out)
        self.out = self.classifier(self.out)
        
        return self.out