import torch
import torch.nn as nn
import torchvision
import numpy as np
from collections import OrderedDict
from models import Contrastive_Loss, Contrastive_Embedder
import torch.nn.functional as F
import math
    
#=    
class Attention(nn.Module):
    def __init__(self, input_size, output_size):
        super(Attention, self).__init__()
        self.output_size = output_size
        self.prompt_layer = nn.Linear(in_features=input_size, out_features=output_size)
        self.key_layer = nn.Linear(in_features=input_size, out_features=output_size)
        self.value_layer = nn.Linear(in_features=input_size, out_features=output_size)
        self.softmax = nn.Softmax()
        
    def forward(self, x):
        out_prompt = F.relu(self.prompt_layer(x))
        out_key = F.relu(self.key_layer(x))
        out_value = F.relu(self.value_layer(x))
        
        out_q_k = torch.div(torch.bmm(out_prompt, out_key.transpose(1, 2)), math.sqrt(self.output_size))
        softmax_q_k = self.softmax(out_q_k)
        out_combine = torch.bmm(softmax_q_k, out_value)
        
        return out_combine

class CNN_Attention(nn.Module):
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
        
        # if(backbone):
        #     self.backbone = nn.Sequential(*backbone)
            
    def forward(self, x):
        self.out = self.conv1(x)
        self.out = self.conv2(self.out)
        self.out = self.maxpool(self.out)
        self.out = self.flatten(self.out)
        self.out = self.fc1(self.out)
        self.out = self.fc2(self.out)
        self.out = self.classifier(self.out)
        
        return self.out

    
class Contrastive_Network_Attention(nn.Module):
    def __init__(self, img_dim=640, backbone=None,num_classes=26, num_kernels=3, input_filter=3, num_filters1=128, num_filters2=64, num_hidden=512, num_hidden2=256, pooling_dim=2, stride=1, padding=1, stridepool=2, paddingpool=0, margin=2.0, embedding_dim=64):
        super().__init__()
        self.backbone = backbone
        if(self.backbone):
            num_hidden_out=self.backbone.fc2.out_features
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            self.embedder = self.embedder=Contrastive_Embedder(num_hidden=num_hidden_out, embedding_dim=embedding_dim)
        else:
            flattendim = int((img_dim/stridepool) ** 2) * num_filters2
            self.extractor = nn.Sequential(
                nn.Conv2d(input_filter, num_filters1, kernel_size=num_kernels, stride=stride, padding=padding),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_filters1, num_filters2, kernel_size=num_kernels, stride=stride, padding=padding),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=pooling_dim, stride=stridepool, padding=paddingpool)
            )
        
            self.flatten= nn.Flatten()
            self.fc = nn.Sequential(
                nn.Linear(flattendim, num_hidden),
                nn.Linear(num_hidden, num_hidden2),
            )
            self.embedder=Contrastive_Embedder(num_hidden=num_hidden2, embedding_dim=embedding_dim)
    
        
# device = 'cpu'
# y1 = torch.tensor([[1.0, 2.0, 3.0],
#                  [3.0, 4.0, 5.0]], dtype=torch.float32).to(device)

# y2 = torch.tensor([[1.0, 2.0, 3.0],
#                  [3.0, 4.0, 5.0]], dtype=torch.float32).to(device)

# loss_func = Contrastive_Loss(2.0)

# loss = loss_func(y1,y2,1)
# print(loss)

# loss = loss_func(y1,y2,0)
# print(loss)