import os
import torch
import torch.nn as nn
import torchvision
from logger import Logger
from dataset import ASL_Dataset
from dataset_bb import ASL_BB_Dataset
from models import BaselineCNN
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from utils import parse_arguments, read_settings, save_checkpoint, load_checkpoint
import numpy as np

# os.environ['https_proxy'] = "http://hpc-proxy00.city.ac.uk:3128"
device = torch.device('cpu')
if torch.cuda.is_available():
   device = torch.device('cuda')
print('cuda' if torch.cuda.is_available() else 'cpu')

model_name = 'BaselineCNN'

def evaluate(logger, data_settings, model, dataloader, mode='Training'):
    model.eval()
    # num_outputs = data_settings['num_output']
    true_labels = []
    predicted_labels = []
    
    for X,y in dataloader:
        X, y = X.to(device), y.to(device)
        ypred = model(X)
        
        ypred = torch.argmax(ypred, axis=1, keepdims=False)

        true_labels.extend(y.cpu().numpy())
        predicted_labels.extend(ypred.cpu().numpy())

    # print(ypred, y)
    overall_accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    
    logger.log({f"{mode} Overall Accuracy": overall_accuracy,
                f"{mode} Precision": precision,
                f"{mode} Recall": recall,
    })
    print("Overall accuracy = {0:.4f}, precision = {1:.4f}, recall={2:.4f}".format(overall_accuracy, precision, recall))
    # print(true_labels[:32],predicted_labels[:32])
    return overall_accuracy, precision

def train(data_settings, model_settings, train_settings):
    
    asl_dataset = ASL_Dataset(mode='train', img_size=data_settings['img_size'])
    asl_bb_dataset = ASL_BB_Dataset(mode='train', img_size=data_settings['img_size'])
    data_len = len(asl_dataset)
    train_len = int(data_len*data_settings['train_size'])
    test_len = int((data_len - train_len)/2)
    val_len = data_len - train_len - test_len
    
    data_len_bb = len(asl_bb_dataset)
    train_len_bb = int(data_len_bb*data_settings['train_size'])
    test_len_bb = int((data_len_bb - train_len_bb)/2)
    val_len_bb = data_len_bb - train_len_bb - test_len_bb
    
    asl_dataset_train, asl_dataset_test, asl_dataset_valid = random_split(asl_dataset, [train_len, test_len, val_len])
    asl_trainloader = DataLoader(asl_dataset_train, batch_size=data_settings['batch_size'], shuffle=True)
    asl_testloader = DataLoader(asl_dataset_test, batch_size=1, shuffle=False)
    asl_validloader = DataLoader(asl_dataset_valid, batch_size=1, shuffle=False)
    asl_bb_loader = DataLoader(asl_bb_dataset, batch_size=1, shuffle=False)
    
    asl_bb_dataset_train, asl_bb_dataset_test, asl_bb_dataset_valid = random_split(asl_bb_dataset, [train_len_bb, test_len_bb, val_len_bb])
    asl_bb_trainloader = DataLoader(asl_bb_dataset, batch_size=data_settings['batch_size'], shuffle=True)
    asl_bb_testloader = DataLoader(asl_bb_dataset_test, batch_size=1, shuffle=False)
    asl_bb_validloader = DataLoader(asl_bb_dataset_valid, batch_size=1, shuffle=False)
    asl_loader = DataLoader(asl_dataset, batch_size=1, shuffle=False)
    
    baselinemodel = BaselineCNN(img_dim=data_settings['img_size'], num_classes=data_settings['num_output'], num_kernels=model_settings['num_kernels'],
                                num_filters1=model_settings['num_filters1'], num_filters2=model_settings['num_filters2'], num_hidden=model_settings['num_hidden'],
                                num_hidden2=model_settings['num_hidden2'], pooling_dim=model_settings['pooling_dim'], stride=model_settings['stride'], padding=model_settings['padding'],
                                stridepool=model_settings['stridepool'], paddingpool=model_settings['paddingpool'])
    ckptfile = f"{model_name}_ckpt.pth"
    if os.path.exists(ckptfile):
        load_checkpoint(baselinemodel, optimizer, ckptfile)
        
    baselinemodel = baselinemodel.to(device)    
    baselinemodel.train()
    
    
    optimizer = torch.optim.Adam(list(baselinemodel.parameters()), lr = train_settings['learning_rate'])
    wandb_logger = Logger(f"inm705_Baseline_CNN", project='inm705_CW')
    logger = wandb_logger.get_logger()
    max_test_acc = 0
    max_val_acc = 0
    for epoch in range(train_settings['epochs']):
        total_loss = 0
        baselinemodel.train()
        
        # for iter,(X,y) in enumerate(asl_trainloader):
        #     optimizer.zero_grad()
        #     X, y = X.to(device), y.to(device)
        #     ypred = baselinemodel(X)

        #     loss = F.cross_entropy(ypred, y.long())
        #     # print(loss)
        #     loss.backward()
        #     optimizer.step()
        #     total_loss+=loss
            
        for iter,(X,y) in enumerate(asl_bb_trainloader):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            ypred = baselinemodel(X)

            loss = F.cross_entropy(ypred, y.long())
            # print(loss)
            loss.backward()
            optimizer.step()
            total_loss+=loss
            
        logger.log({'train_loss': total_loss/len(asl_trainloader)})
        print('Epoch:{}, Train Loss:{}'.format(epoch, total_loss/len(asl_trainloader)))
        
        # train_acc, train_prec = evaluate(logger,data_settings,baselinemodel,asl_trainloader, mode='Training')
        # test_acc, test_prec = evaluate(logger,data_settings,baselinemodel,asl_testloader, mode='Testing')
        # val_acc, val_prec = evaluate(logger,data_settings,baselinemodel,asl_validloader, mode='Validation')
        # bb_acc, bb_prec = evaluate(logger,data_settings,baselinemodel,asl_bb_trainloader, mode='Testing BB')
        
        train_bb_acc, train_bb_prec = evaluate(logger,data_settings,baselinemodel,asl_bb_trainloader, mode='Training BB')
        test_bb_acc, test_bb_prec = evaluate(logger,data_settings,baselinemodel,asl_bb_testloader, mode='Testing BB')
        val_bb_acc, val_bb_prec = evaluate(logger,data_settings,baselinemodel,asl_bb_validloader, mode='Validation BB')
        ori_acc, ori_prec = evaluate(logger,data_settings,baselinemodel,asl_trainloader, mode='Testing Ori')
        
        # if((test_acc > max_test_acc) and (val_acc > max_val_acc)):
        #     save_checkpoint(epoch, baselinemodel, 'BaselineCNN', optimizer)
        #     max_test_acc = test_acc
        #     max_val_acc = val_acc
            
        if((test_bb_acc > max_test_acc) and (val_bb_acc > max_val_acc)):
            save_checkpoint(baselinemodel, 'BaselineCNN', optimizer)
            max_test_acc = test_bb_acc
            max_val_acc = val_bb_acc
    return

def main():
    args = parse_arguments()

    # Read settings from the YAML file
    filepath=os.path.dirname(os.path.realpath(__file__))
    settings = read_settings(filepath+args.config)

    # Access and use the settings as needed
    data_settings = settings.get('data', {})
    model_settings = settings.get('model', {})
    train_settings = settings.get('train', {})

    train(data_settings, model_settings, train_settings)
    
    
if __name__ == '__main__':
    main()