import os
import zipfile
import torch
import torch.utils.data as data
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from PIL import Image
from PIL import Image, ImageDraw
import torch.nn as nn
import random
# import datasets

# Conventional dataset structure for ASL_dataset
class ASL_Dataset(data.Dataset):
    def __init__(self, mode='train', filename='archive', img_size=640, include_others=False):
        super(ASL_Dataset, self).__init__()
        
        #Initialize variables
        self.filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)
        imgpath="asl_alphabet_{}/asl_alphabet_{}".format(mode,mode) if mode=='train' else "asl_alphabet_{}".format(mode)
        
        self.filedir=os.path.join(self.filepath,imgpath)
        self.img_size = img_size
        self.class_dict={}
        self.inv_class_dict={}
        self.imagelist = []
        self.labellist = []
        self.class_per_dict = 3000
        
        # Add original 26 classes
        alphabetlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for iter,clas in enumerate(alphabetlist):
            self.class_dict.update({clas:iter})
            self.inv_class_dict.update({iter:clas})
        temp_num_classes = len(self.class_dict.keys())
        
        # Add addtional 3 classes only if prompted in config
        if(include_others):
            for clas in os.listdir(self.filedir):
                if(clas not in alphabetlist):
                    self.class_dict.update({clas:temp_num_classes})
                    temp_num_classes+=1
                
                
        # access files in directory and add to list
        for tempclass in os.listdir(self.filedir):
            if(not include_others):
                if tempclass in alphabetlist:
                    classpath = os.path.join(self.filedir,tempclass)
                    for imgname in os.listdir(classpath):
                        self.imagelist.append(os.path.join(classpath,imgname))
                        self.labellist.append(tempclass)
            else:
                classpath = os.path.join(self.filedir,tempclass)
                for imgname in os.listdir(classpath):
                    self.imagelist.append(os.path.join(classpath,imgname))
                    self.labellist.append(tempclass)
         
        # apply transforms
        self.transform = transforms.Compose([
        transforms.Resize(self.img_size),
        transforms.CenterCrop(self.img_size),
        transforms.ToTensor(),
        ])
        
    # Loads image with transforms
    def load_img(self,index):
        im = Image.open(self.imagelist[index])
        im = self.transform(im)
        
        return im
        
    # Gets length of dataset
    def __len__(self):
        return len(self.imagelist)
    
    # Retrieves indexed datapoint
    def __getitem__(self, index):
        im = self.load_img(index)
        lab = torch.as_tensor(self.class_dict[self.labellist[index]], dtype=torch.int64)
        
        
        return im, lab
    
    # For visualisation purposes, not used in final code
    def visualise(self,index):
        im = Image.open(self.imagelist[index])
        print('Current label is: ', self.labellist[index])
        im.show()
        return

# Contrastive Datasets, using the same dataset from ASL_Dataset with changes
class ASL_Dataset_Contrastive(data.Dataset):
    def __init__(self, mode='train', filename='archive', img_size=640, include_others=False, simi_ratio=0.5):
        super(ASL_Dataset_Contrastive, self).__init__()
        
        #Initialize variables
        self.filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)
        imgpath="asl_alphabet_{}/asl_alphabet_{}".format(mode,mode)
        
        self.filedir=os.path.join(self.filepath,imgpath)
        self.img_size = img_size
        self.class_dict={}
        self.inv_class_dict={}
        self.imagelist = []
        self.labellist = []
        self.simi_ratio = simi_ratio
        self.class_per_dict = 3000 if mode=='train' else 30
        
        alphabetlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for iter,clas in enumerate(alphabetlist):
            self.class_dict.update({clas:iter})
            self.inv_class_dict.update({iter:clas})
        temp_num_classes = len(self.class_dict.keys())
        
        if(include_others):
            for clas in os.listdir(self.filedir):
                if(clas not in alphabetlist):
                    self.class_dict.update({clas:temp_num_classes})
                    temp_num_classes+=1
                
                
                
        for tempclass in os.listdir(self.filedir):
            if(not include_others):
                if tempclass in alphabetlist:
                    classpath = os.path.join(self.filedir,tempclass)
                    for imgname in os.listdir(classpath):
                        self.imagelist.append(os.path.join(classpath,imgname))
                        self.labellist.append(tempclass)
            else:
                classpath = os.path.join(self.filedir,tempclass)
                for imgname in os.listdir(classpath):
                    self.imagelist.append(os.path.join(classpath,imgname))
                    self.labellist.append(tempclass)
         
        self.transform = transforms.Compose([
        transforms.Resize(self.img_size),
        transforms.CenterCrop(self.img_size),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.457, 0.407],
        #                     std=[0.224, 0.224, 0.225] )
        ])
        
    # Loads image with transforms
    def load_img(self,index):
        im = Image.open(self.imagelist[index])
        im = self.transform(im)
        
        return im
        
    # Gets length of dataset
    def __len__(self):
        return len(self.imagelist)
    
    # Retrieves indexed datapoint
    def __getitem__(self, index):
        im1 = self.load_img(index)
        lab1 = torch.as_tensor(self.class_dict[self.labellist[index]], dtype=torch.int64)
        
        
        #function to increase frequency of positive pairings with 50% probability
        if random.random() < self.simi_ratio: 
            
            random_index = lab1*self.class_per_dict #Takes the first label of the class as anchor
            
            #Ensures no repeated datapoints
            while random_index == index: 
                random_index = random.randint(lab1 * self.class_per_dict, (lab1+1) * self.class_per_dict -1)

            im2 = self.load_img(random_index)
            lab2 = torch.as_tensor(self.class_dict[self.labellist[random_index]], dtype=torch.int64)

            simi = int(torch.equal(lab1, lab2))
        else: 
            # Takes random anchor from the dataset, possibly takes only the first label of each class.
            random_index = random.randint(0,len(self.class_dict)-1)* self.class_per_dict
            im2 = self.load_img(random_index)
            lab2 = torch.as_tensor(self.class_dict[self.labellist[random_index]], dtype=torch.int64)
            simi = int(torch.equal(lab1, lab2))
            
            #Ensures no repeated datapoints
            while random_index == index: 
                random_index = random.randint(0,len(self))
            
                im2 = self.load_img(random_index)
                lab2 = torch.as_tensor(self.class_dict[self.labellist[random_index]], dtype=torch.int64)
                
                simi = int(torch.equal(lab1, lab2))
        
        #im1: data of indexed image
        #im2: data of random anchor datapoint
        #simi: binary value describing the similarity of class between im1 and im2
        #lab1: label of im1, for plotting purposes
        return im1, im2, simi, lab1
    
    # For visualization purposes only, not used in final code.
    def visualise(self,index):
        im = Image.open(self.imagelist[index])
        print('Current label is: ', self.labellist[index])
        im.show()
        return
