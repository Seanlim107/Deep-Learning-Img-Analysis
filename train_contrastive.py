import os
import torch
import torch.nn as nn
import torchvision
from logger import Logger
from datasets import ASL_Dataset, ASL_BB_Dataset, ASL_C_Dataset, ASL_Dataset_Contrastive
from models import BaselineCNN, Contrastive_Network, Contrastive_Loss, Contrastive_Embedder
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from utils import parse_arguments, read_settings, save_checkpoint, load_checkpoint
import numpy as np
from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans
# import pandas as pd
import seaborn as sns

# os.environ['https_proxy'] = "http://hpc-proxy00.city.ac.uk:3128"
device = torch.device('cpu')
if torch.cuda.is_available():
   device = torch.device('cuda')
print('cuda' if torch.cuda.is_available() else 'cpu')

model_name = 'BaselineCNN'

def plot_embeds(data_settings, model, dataloader, mode='Training', logger=None):
    # Preparations for evaluation
    model.eval()
    embeddings=[]
    labels=[]
    tsne = TSNE(n_components=3, random_state=42)
    
    # Evaluate predictions with true outputs
    with torch.no_grad():
        for X1, X2, y, y1 in dataloader:
            X1, X2, y = X1.to(device), X2.to(device), y.to(device)
            ypred1, _ = model(X1, X2)
            labels.append(y.cpu().numpy())
            embeddings.append(ypred1.cpu().numpy())
    
    to_plot = tsne(embeddings)
    ax = sns.scatter(to_plot[:,0], to_plot[:,1], to_plot[:,2], label=labels)
    
        

    # Calculate recall and precision
    # overall_accuracy = accuracy_score(true_labels, predicted_labels)
    # precision = precision_score(true_labels, predicted_labels, average='weighted')
    # recall = recall_score(true_labels, predicted_labels, average='weighted')
    
    # _____________________________________________________________ TURN OFF FOR DEBUGGING __________________________________________________________________________
    logger.log({f"{mode} Precision": precision,
                f"{mode} Recall": recall,
    })
    # _____________________________________________________________ TURN OFF FOR DEBUGGING __________________________________________________________________________
    
    print("Overall accuracy = {0:.4f}, precision = {1:.4f}, recall={2:.4f}".format(overall_accuracy, precision, recall))
    # print(true_labels[:32],predicted_labels[:32])
    return overall_accuracy, precision

def train(data_settings, model_settings, train_settings):
    
    # asl_dataset: original dataset with 87k datapoints
    # asl_bb_dataset: dataset with 3k datapoints, 26 classes with bounding box
    # asl_c_dataset: dataset with 900 datapoints, to be used for contrastive learning
    
    asl_dataset = ASL_Dataset_Contrastive(mode='train', img_size=data_settings['img_size'])
    # asl_bb_dataset = ASL_BB_Dataset(mode='train', img_size=data_settings['img_size'], method=None)
    # asl_c_dataset = ASL_C_Dataset(mode='train', img_size=data_settings['img_size'])
    
    # Split datapoints
    data_len = len(asl_dataset)
    train_len = int(data_len*data_settings['train_size'])
    test_len = int((data_len - train_len)/2)
    val_len = data_len - train_len - test_len

    
    asl_dataset_train, asl_dataset_test, asl_dataset_valid = random_split(asl_dataset, [train_len, test_len, val_len])
    asl_trainloader = DataLoader(asl_dataset_train, batch_size=data_settings['batch_size'], shuffle=True)
    asl_testloader = DataLoader(asl_dataset_test, batch_size=1, shuffle=False)
    asl_validloader = DataLoader(asl_dataset_valid, batch_size=1, shuffle=False)
    
    # Baseline model for evaluating
    baselinemodel = BaselineCNN(img_dim=data_settings['img_size'], num_classes=data_settings['num_output'], num_kernels=model_settings['num_kernels'],
                                num_filters1=model_settings['num_filters1'], num_filters2=model_settings['num_filters2'], num_hidden=model_settings['num_hidden'],
                                num_hidden2=model_settings['num_hidden2'], pooling_dim=model_settings['pooling_dim'], stride=model_settings['stride'], padding=model_settings['padding'],
                                stridepool=model_settings['stridepool'], paddingpool=model_settings['paddingpool'])
    # 
    
    
    ckptfile = f"{model_name}_ckpt.pth"
    optimizer = torch.optim.Adam(list(baselinemodel.parameters()), lr = train_settings['learning_rate'])
    loss_func = Contrastive_Loss(margin=2.0)
    # Load checkpoint if possible
    if os.path.exists(ckptfile):
        load_checkpoint(baselinemodel, optimizer, ckptfile)
        contrastive_model_pretrained = Contrastive_Network(backbone=baselinemodel)
        contrastive_model = contrastive_model_pretrained.to(device)
    else:
        contrastive_baselinemodel = Contrastive_Network(img_dim=data_settings['img_size'], num_classes=data_settings['num_output'], num_kernels=model_settings['num_kernels'],
        num_filters1=model_settings['num_filters1'], num_filters2=model_settings['num_filters2'], num_hidden=model_settings['num_hidden'],
        num_hidden2=model_settings['num_hidden2'], pooling_dim=model_settings['pooling_dim'], stride=model_settings['stride'], padding=model_settings['padding'],
        stridepool=model_settings['stridepool'], paddingpool=model_settings['paddingpool'], embedding_dim=model_settings['embedding_dim'])
        contrastive_model = contrastive_baselinemodel.to(device)
    
    tsne = TSNE(n_components=3, random_state=1, perplexity=30, metric="cosine")
    
    
    # Initialise wandb logger
    # _____________________________________________________________ TURN OFF FOR DEBUGGING __________________________________________________________________________
    wandb_logger = Logger(f"inm705_Baseline_Contrastive", project='inm705_CW')
    logger = wandb_logger.get_logger()
    # _____________________________________________________________ TURN OFF FOR DEBUGGING __________________________________________________________________________
    
    # variables for checkpoint saving
    max_test_acc = 0
    max_val_acc = 0
    
    # Training loop per epoch
    for epoch in range(train_settings['epochs']):
        total_loss = 0
        baselinemodel.train()
        
        for iter,(X1,X2,simi, y1) in enumerate(asl_trainloader):
            optimizer.zero_grad()
            X1, X2, simi = X1.to(device), X2.to(device), simi.to(device)
            ypred1, ypred2= contrastive_model(X1,X2)
            # print(ypred1.size())
            loss = loss_func(ypred1, ypred2, simi.long())

            # # print(loss)
            loss.backward()
            optimizer.step()
            total_loss+=loss
            
        # _____________________________________________________________ TURN OFF FOR DEBUGGING __________________________________________________________________________
            logger.log({'train_loss': total_loss/len(asl_trainloader)})
            print('Epoch:{}, Train Loss:{}'.format(epoch, total_loss/len(asl_trainloader)))
        # _____________________________________________________________ TURN OFF FOR DEBUGGING __________________________________________________________________________
        
        plot_embeds(data_settings, contrastive_model, asl_trainloader, mode='Training', logger=None)
        plot_embeds(data_settings, contrastive_model, asl_testloader, mode='Training', logger=None)
        plot_embeds(data_settings, contrastive_model, asl_validloader, mode='Training', logger=None)
        # train_acc, train_prec = evaluate(data_settings,baselinemodel,asl_trainloader, mode='Training')
        # test_acc, test_prec = evaluate(data_settings,baselinemodel,asl_testloader, mode='Testing')
        # val_acc, val_prec = evaluate(data_settings,baselinemodel,asl_validloader, mode='Validation')

        # if((test_acc > max_test_acc) and (val_acc > max_val_acc)):
        #     save_checkpoint(epoch, baselinemodel, 'BaselineCNN', optimizer)
        #     max_test_acc = test_acc
        #     max_val_acc = val_acc

    return

def main():
    args = parse_arguments()
    # print(args)
    # Read settings from the YAML file
    settings = read_settings(args.config)

    # Access and use the settings as needed
    data_settings = settings.get('data', {})
    model_settings = settings.get('model', {})
    train_settings = settings.get('train', {})

    train(data_settings, model_settings, train_settings)
    
    
if __name__ == '__main__':
    main()