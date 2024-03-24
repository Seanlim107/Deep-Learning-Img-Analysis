import os
import zipfile
import torch
import torch.utils.data as data
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from PIL import Image
# import datasets


class ASL_Dataset(data.Dataset):
    def __init__(self, mode='train', filename='archive', img_size=640):
        super(ASL_Dataset, self).__init__()
        
        #Initialize variablesd
        self.filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)
        imgpath="asl_alphabet_{}/asl_alphabet_{}".format(mode,mode)
        self.filedir=os.path.join(self.filepath,imgpath)
        self.img_size = img_size
        self.class_dict={}
        self.inv_class_dict={}
        self.imagelist = []
        self.labellist = []
        
        alphabetlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for iter,clas in enumerate(alphabetlist):
            self.class_dict.update({clas:iter})
            self.inv_class_dict.update({iter:clas})
        temp_num_classes = len(self.class_dict.keys())
        
        for clas in os.listdir(self.filedir):
            if(clas not in alphabetlist):
                self.class_dict.update({clas:temp_num_classes})
                
                temp_num_classes+=1
                
                
                
        for tempclass in os.listdir(self.filedir):
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
        return self.load_img(index), torch.as_tensor(self.class_dict[self.labellist[index]], dtype=torch.int64)
    
    def visualise(self,index):
        im = Image.open(self.imagelist[index])
        print('Current label is: ', self.labellist[index])
        im.show()
        return
    
my_asl = ASL_Dataset()
X,y = my_asl[0]
# print(my_asl.inv_class_dict[y.item()])
print(X.size())