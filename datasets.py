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

class ASL_Dataset(data.Dataset):
    def __init__(self, mode='train', filename='archive', img_size=640, include_others=False):
        super(ASL_Dataset, self).__init__()
        
        #Initialize variablesd
        self.filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)
        imgpath="asl_alphabet_{}/asl_alphabet_{}".format(mode,mode) if mode=='train' else "asl_alphabet_{}".format(mode)
        
        self.filedir=os.path.join(self.filepath,imgpath)
        self.img_size = img_size
        self.class_dict={}
        self.inv_class_dict={}
        self.imagelist = []
        self.labellist = []
        self.class_per_dict = 3000
        
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
        transforms.Normalize(mean=[0.485, 0.457, 0.407],
                            std=[0.224, 0.224, 0.225] )
        ])
        
    def load_img(self,index):
        im = Image.open(self.imagelist[index])
        im = self.transform(im)
        
        return im
        
    def __len__(self):
        return len(self.imagelist)
    
    def __getitem__(self, index):
        im = self.load_img(index)
        lab = torch.as_tensor(self.class_dict[self.labellist[index]], dtype=torch.int64)
        
        
        return im, lab
    
    def visualise(self,index):
        im = Image.open(self.imagelist[index])
        print('Current label is: ', self.labellist[index])
        im.show()
        return

class ASL_Dataset_Contrastive(data.Dataset):
    def __init__(self, mode='train', filename='archive', img_size=640, include_others=False):
        super(ASL_Dataset_Contrastive, self).__init__()
        
        #Initialize variablesd
        self.filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)
        imgpath="asl_alphabet_{}/asl_alphabet_{}".format(mode,mode) if mode=='train' else "asl_alphabet_{}".format(mode)
        
        self.filedir=os.path.join(self.filepath,imgpath)
        self.img_size = img_size
        self.class_dict={}
        self.inv_class_dict={}
        self.imagelist = []
        self.labellist = []
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
        transforms.Normalize(mean=[0.485, 0.457, 0.407],
                            std=[0.224, 0.224, 0.225] )
        ])
        
    def load_img(self,index):
        im = Image.open(self.imagelist[index])
        im = self.transform(im)
        
        return im
        
    def __len__(self):
        return len(self.imagelist)
    
    def __getitem__(self, index):
        im1 = self.load_img(index)
        lab1 = torch.as_tensor(self.class_dict[self.labellist[index]], dtype=torch.int64)
        
        
        #function to increasae frequency of positive pairings with 50% probability
        if random.random() < 0.45: 
            
            random_index = random.randint(lab1 * self.class_per_dict, (lab1+1) * self.class_per_dict -1)
            while random_index == index: #Ensures no repeated datapoints
                random_index = random.randint(lab1 * self.class_per_dict, (lab1+1) * self.class_per_dict -1)
            # print(random_index)
            im2 = self.load_img(random_index)
            lab2 = torch.as_tensor(self.class_dict[self.labellist[random_index]], dtype=torch.int64)
            
            simi = int(torch.equal(lab1, lab2))
        else: 
            random_index = random.randint(0,len(self)-1)
            im2 = self.load_img(random_index)
            lab2 = torch.as_tensor(self.class_dict[self.labellist[random_index]], dtype=torch.int64)
            
            simi = int(torch.equal(lab1, lab2))
            while random_index == index: #Ensures no repeated datapoints
                random_index = random.randint(0,len(self))
            
                im2 = self.load_img(random_index)
                lab2 = torch.as_tensor(self.class_dict[self.labellist[random_index]], dtype=torch.int64)
                
                simi = int(torch.equal(lab1, lab2))
        
        return im1, im2, simi, lab1
    
    def visualise(self,index):
        im = Image.open(self.imagelist[index])
        print('Current label is: ', self.labellist[index])
        im.show()
        return
    

class ASL_C_Dataset(data.Dataset):
    def __init__(self, mode='train', filename='archive_c', img_size=640):
        super(ASL_C_Dataset, self).__init__()
        
        #Initialize variablesd
        self.filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)
        # imgpath=self.filepath
        self.filedir=self.filepath
        self.img_size = img_size
        self.class_dict={}
        self.inv_class_dict={}
        self.imagelist = []
        self.labellist = []
        self.zsl_list = []
        
        alphabetlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for iter,clas in enumerate(alphabetlist):
            self.class_dict.update({clas:iter})
            self.inv_class_dict.update({iter:clas})
        temp_num_classes = len(self.class_dict.keys())
        
        # Add classes that are not alphabets
        for clas in os.listdir(self.filedir):
            if(clas not in alphabetlist):
                # if(clas=='asl-alphabet-test'):
                #     continue
                # else:
                self.class_dict.update({clas:temp_num_classes})
                temp_num_classes+=1
                classpath = os.path.join(self.filedir,clas)
                for imgname in os.listdir(classpath):
                    self.imagelist.append(os.path.join(classpath,imgname))
                    self.labellist.append(clas)
                
         
        self.transform = transforms.Compose([
        transforms.Resize((self.img_size,self.img_size)),
        transforms.CenterCrop(self.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.457, 0.407],
                            std=[0.224, 0.224, 0.225] )
        ])
        
    def load_img(self,index):
        im = Image.open(self.imagelist[index])
        im = self.transform(im)
        
        return im
        
    def __len__(self):
        return len(self.imagelist)
    
    def __getitem__(self, index):
        return self.load_img(index), torch.as_tensor(self.class_dict[self.labellist[index]], dtype=torch.int64)
    
    def visualise(self,index):
        im = Image.open(self.imagelist[index])
        print('Current label is: ', self.labellist[index])
        im.show()
        return
    

class ASL_BB_Dataset(data.Dataset):
    def __init__(self, mode='train', transform=None, num_classes=None, filename=None, imgpath=None, img_size=640, method=None):
        super(ASL_BB_Dataset, self).__init__()
        self.num_classes = num_classes
        self.filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'archive_bb')
        self.filedirlist = ['train', 'test', 'valid']
        self.img_size = img_size
        self.img_path = os.path.join(mode,'images')
        self.label_path = os.path.join(mode,'labels')
        self.method = method
        self.mode=mode
        
        if((self.method!='yolo') and (self.method!='rcnn') and (self.method!=None)):
            raise Exception('Invalid model method made')
        
        if mode not in self.filedirlist:
            raise Exception('Acceptable modes are only train, test and valid')
        
        self.image_files=sorted(os.listdir(os.path.join(self.filepath, self.img_path)))
        self.label_files=sorted(os.listdir(os.path.join(self.filepath, self.label_path)))
        
        self.inv_class_dict = {}
        alphabetlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for iter,clas in enumerate(alphabetlist):
            self.inv_class_dict.update({iter:clas})
        

        self.transform = transforms.Compose(
            [
        transforms.Resize((self.img_size,self.img_size)),
        transforms.CenterCrop(self.img_size),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.457, 0.407],
                            std=[0.224, 0.224, 0.225] )
            ]
                                                )
        
    def __len__(self):
        return len(self.image_files)
    
    def extract_bb_class(self, index):
        label_name = open(os.path.join(self.filepath, self.label_path, self.label_files[index]),'r')
        label_name = label_name.read().split(' ')
        indexed_class = int(label_name[0])
        
        bb_coords = [int(float(x)*self.img_size) for x in label_name[1:]]
        # print(bb_coords)
        out={}
        if(self.method=='yolo'):
            out['class'] = torch.as_tensor(indexed_class, dtype=torch.int64)
            out['box'] = torch.as_tensor(bb_coords)
        elif(self.method=='rcnn'):
            xmin=bb_coords[0]-int(bb_coords[2]/2)
            xmax=bb_coords[0]+int(bb_coords[2]/2)
            ymin=bb_coords[1]-int(bb_coords[3]/2)
            ymax=bb_coords[1]+int(bb_coords[3]/2)
            out['class'] = torch.as_tensor(indexed_class, dtype=torch.int64)
            out['box'] = torch.as_tensor([xmin, ymin, xmax, ymax])
        elif(self.method==None):
            out = torch.as_tensor(indexed_class, dtype=torch.int64)
            
        return out
    
    def load_img(self,index):
        img_name = os.path.join(self.filepath, self.img_path, self.image_files[index])
        im = Image.open(img_name)
        if(self.mode=='test' or self.mode=='valid'):
            self.transform = transforms.Compose(
            [
        transforms.Resize((self.img_size,self.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.457, 0.407],
                            std=[0.224, 0.224, 0.225] )
            ]
                                                )
        im = self.transform(im)
        
        return im
    
    def __getitem__(self, index):
        image = self.load_img(index)
        label = self.extract_bb_class(index)
        
        return image,label
    
    def visualise(self, index):
        img_name = os.path.join(self.filepath, self.img_path, self.image_files[index])
        im = Image.open(img_name)
        _,label = self[index]
        draw = ImageDraw.Draw(im)
        toplot=list(label['box'])
        if(self.method=='rcnn'):
            draw.rectangle(toplot, fill='black')
        elif(self.method=='yolo'):
            toplot
            xmin=toplot[0]-int(toplot[2]/2)
            xmax=toplot[0]+int(toplot[2]/2)
            ymin=toplot[1]-int(toplot[3]/2)
            ymax=toplot[1]+int(toplot[3]/2)
            draw.rectangle([xmin,ymin,xmax,ymax], fill='black')
        print('label is: ', self.inv_class_dict[int(label['class'])])
        im.show()
        
        return
    
    
# from torch.utils.data import random_split, DataLoader
# asl_dataset = ASL_Dataset_Contrastive(mode='train', img_size=320)
# data_len = len(asl_dataset)
# train_len = int(data_len*0.8)
# test_len = int((data_len - train_len)/2)
# val_len = data_len - train_len - test_len


# asl_dataset_train, asl_dataset_test, asl_dataset_valid = random_split(asl_dataset, [train_len, test_len, val_len])
# asl_trainloader = DataLoader(asl_dataset_train, batch_size=16, shuffle=True)
# asl_train_testloader = DataLoader(asl_dataset_train, batch_size=1, shuffle=False)
# asl_testloader = DataLoader(asl_dataset_test, batch_size=1, shuffle=False)
# asl_validloader = DataLoader(asl_dataset_valid, batch_size=1, shuffle=False)

# for iter,(x1,x2,y,y1) in enumerate(asl_trainloader):
#     # print(y)
#     continue

# for iter,(x1,x2,y,y1) in enumerate(asl_train_testloader):
#     # print(y)
#     continue

# for iter,(x1,x2,y,y1) in enumerate(asl_testloader):
#     # print(y)
#     continue

# for iter,(x1,x2,y,y1) in enumerate(asl_validloader):
#     # print(y)
    # continue