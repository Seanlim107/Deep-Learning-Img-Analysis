import os
import zipfile
from zipfile import ZipFile
import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader
# import datasets

# train_data = torchvision.datasets.ImageFolder(filedir, transform=my_transform)
# train_data_loader = data.DataLoader(train_data, batch_size=16, shuffle=True,  num_workers=0)

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
        
        self.transform = transforms.Compose([
        transforms.Resize(self.img_size),
        transforms.CenterCrop(self.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.457, 0.407],
                            std=[0.224, 0.224, 0.225] )
        ])
        
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
        
# testASL = ASL_BB_Dataset(mode='train', method=None)
# x,y = testASL[0]
# print(y)
# print(torch.Tensor.size(x))
# # print(testASL[0])
# testASL.visualise(10)