import torch
import torch.nn as nn


############################################################################ Baseline Convolutional Models ##########################################################################################################

class BaselineCNN(nn.Module):
    def __init__(self, img_dim=640, backbone=None,num_classes=29, num_kernels=3, input_filter=3, num_filters1=128, num_filters2=64, num_hidden=512, num_hidden2=256, pooling_dim=2, stride=1, padding=1, stridepool=2, paddingpool=0):
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
    
############################################################################ Contrastive Models ##########################################################################################################
            
            
# Function to embed datapoints
class Contrastive_Embedder(nn.Module):
    def __init__(self, num_hidden=64, embedding_dim=64):
        super(Contrastive_Embedder, self).__init__()
        self.embedder = nn.Linear(num_hidden, embedding_dim)
        
    def forward(self, x):
        embedded = self.embedder(x)
        
        return embedded
    
# Contrastive Loss function, inspired from https://jamesmccaffrey.wordpress.com/2022/03/04/contrastive-loss-function-in-pytorch/
class Contrastive_Loss(nn.Module):
    def __init__(self, margin=2.0):
        super(Contrastive_Loss, self).__init__()
        self.margin = margin
            
    def forward(self, y1, y2, target_pair):
        #target pair indicates a label if the pair should be idnetical or dissimilar
        euc_distance = nn.functional.pairwise_distance(y1, y2)
        return torch.mean((target_pair)*(torch.pow(torch.clamp(self.margin - euc_distance, min=0.0), 2)) + (1-target_pair)*(torch.pow(euc_distance, 2)))

        
# Contrastive Network, can be trained from scratch or utilising a backbone file
class Contrastive_Network(nn.Module):
    def __init__(self, img_dim=640, type=0, backbone=None,num_classes=29, num_kernels=3, input_filter=3, num_filters1=128, num_filters2=64, num_hidden=512, num_hidden2=256, pooling_dim=2, stride=1, padding=1, stridepool=2, paddingpool=0, margin=2.0, embedding_dim=64):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        
        if(self.backbone):
            flattendim = int((img_dim/stridepool) ** 2) * num_filters2
            if type==0: # FNCL
                self.backbone = nn.Sequential(*list(self.backbone.children())[:4])
                self.embedder =Contrastive_Embedder(num_hidden=flattendim, embedding_dim=embedding_dim)
            else:  #FCFCL
                self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
                self.embedder =Contrastive_Embedder(num_hidden=num_hidden2, embedding_dim=embedding_dim)
            
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
        
    def forward(self, x1, x2):
        
        if(self.backbone):
            y1=self.backbone(x1)
            y2=self.backbone(x2)
            e1=self.embedder(y1)
            e2=self.embedder(y2)
        else:
            y1 = self.extractor(x1)
            y1 = self.flatten(y1)
            y1 = self.fc(y1)
            e1 = self.embedder(y1)
            
            y2 = self.extractor(x2)
            y2 = self.flatten(y2)
            y2 = self.fc(y2)
            e2 = self.embedder(y2)
        return e1,e2

