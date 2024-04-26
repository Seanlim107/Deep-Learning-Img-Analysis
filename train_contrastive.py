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
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from torchvision import utils

# from sklearn.datasets import load_wine
# winedata = load_wine()
# X, ytest = winedata['data'], winedata['target']
# from sklearn.cluster import KMeans
# import pandas as pd

# os.environ['https_proxy'] = "http://hpc-proxy00.city.ac.uk:3128"
device = torch.device('cpu')
if torch.cuda.is_available():
   device = torch.device('cuda')
print('cuda' if torch.cuda.is_available() else 'cpu')

model_name = 'BaselineCNN_ckpt2'
cont_model_name = 'My_Contrastive_No_Backbone'

def plot_embeds(data_settings, model, dataloader, epoch, mode='Training', logger=None, ):
    # Preparations for evaluation
    model.eval()
    embeddings=[]
    labels=[]
    tsne = TSNE(n_components=3, random_state=42)
    pca = PCA(n_components=3, random_state=42)
    fig=plt.figure(figsize=(8,6))
    ax = fig.add_subplot(projection='3d')
    # count = len(model.)
    # Evaluate predictions with true outputs
    with torch.no_grad():
        
        for iter, (X1, X2, y, y1) in enumerate(dataloader):
            X1, X2, y = X1.to(device), X2.to(device), y.to(device)
            ypred1, _ = model(X1, X2)
            labels.append(y1.cpu().detach().numpy())
            embeddings.append(ypred1.cpu().detach().numpy())
        # print(np.array(embeddings).size)
        labels=np.array(labels)
        # print(np.shape(labels))
        embeddings=np.concatenate(embeddings, axis=0)
        cmap = plt.cm.get_cmap('tab20', len(np.unique(labels)))
        # to_plot = tsne.fit_transform(embeddings)
        to_plot_pca = pca.fit_transform(embeddings)
        for label in np.unique(labels):
            class_indices=np.where(labels==label)[0]
            ax.scatter(to_plot_pca[class_indices,0], to_plot_pca[class_indices,1], zs=to_plot_pca[class_indices,2], zdir='z', label=label, c=[cmap(label)])
            # ax.scatter(to_plot_pca[class_indices,0], to_plot_pca[class_indices,1], label=label, c=[cmap(label)])
        # ax.scatter(to_plot_pca[:,0], to_plot_pca[:,1], zs=to_plot_pca[:,2], zdir='z', c=labels[:])
        # plt.show()
        directory='plot_images_3'
        if not os.path.exists(directory):
            os.makedirs(directory)
        # plt.show()
        ax.legend()
        # plt.show()
        plt.savefig(os.path.join(directory,f"{mode}_{epoch}.png"), bbox_inches='tight')
        plt.close()
        
    # _____________________________________________________________ TURN OFF FOR DEBUGGING __________________________________________________________________________
    # logger.log({f"{mode} Precision": precision,
    #             f"{mode} Recall": recall,
    # })
    # _____________________________________________________________ TURN OFF FOR DEBUGGING __________________________________________________________________________

    return

def train(data_settings, model_settings, train_settings):
    
    # asl_dataset: original dataset with 87k datapoints, cut to 26 classes and 78k datapoints
    # asl_bb_dataset: dataset with 3k datapoints, 26 classes with bounding box
    # asl_c_dataset: dataset with 900 datapoints, to be used for contrastive learning
    include_others = False if data_settings['num_output']<=26 else True
    asl_dataset = ASL_Dataset_Contrastive(mode='train', img_size=data_settings['img_size'], simi_ratio=data_settings['simi_ratio'], include_others=include_others)
    # asl_bb_dataset = ASL_BB_Dataset(mode='train', img_size=data_settings['img_size'], method=None)
    # asl_c_dataset = ASL_C_Dataset(mode='train', img_size=data_settings['img_size'])
    
    # Split datapoints
    data_len = len(asl_dataset)
    train_len = int(data_len*data_settings['train_size'])
    test_len = int((data_len - train_len)/2)
    val_len = data_len - train_len - test_len

    
    asl_dataset_train, asl_dataset_test, asl_dataset_valid = random_split(asl_dataset, [train_len, test_len, val_len])
    asl_trainloader = DataLoader(asl_dataset_train, batch_size=data_settings['batch_size'], shuffle=True)
    asl_train_testloader = DataLoader(asl_dataset_train, batch_size=1, shuffle=False)
    asl_testloader = DataLoader(asl_dataset_test, batch_size=1, shuffle=False)
    asl_validloader = DataLoader(asl_dataset_valid, batch_size=1, shuffle=False)
    
    # Baseline model for evaluating
    baselinemodel = BaselineCNN(img_dim=data_settings['img_size'], num_classes=data_settings['num_output'], num_kernels=model_settings['num_kernels'],
                                num_filters1=model_settings['num_filters1'], num_filters2=model_settings['num_filters2'], num_hidden=model_settings['num_hidden'],
                                num_hidden2=model_settings['num_hidden2'], pooling_dim=model_settings['pooling_dim'], stride=model_settings['stride'], padding=model_settings['padding'],
                                stridepool=model_settings['stridepool'], paddingpool=model_settings['paddingpool'])
    # 
    backbonefile = f"{model_name}.pth"
    if os.path.isfile(backbonefile):
        print('Backbone detected, installing backbone')
        backbone_ckpt = torch.load(backbonefile, map_location=device)
        backbone=baselinemodel
        
        backbone.load_state_dict(backbone_ckpt['model_state'])
        contrastive_baselinemodel = Contrastive_Network(backbone=backbone, embedding_dim=model_settings['embedding_dim'])
        # for m in contrastive_baselinemodel.backbone.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n,c,w,h = m.weight.data.shape
        #         # weight_to_plot = m.weight.data.view(n*c,-1,w,h )
        #         weight_to_plot = m.weight.data[:,0,:,:].unsqueeze(dim=1)
        #         nrow=16
        #         rows = np.min( (weight_to_plot.shape[0]//nrow + 1, 64 )  ) 
        #         grid = utils.make_grid(weight_to_plot, nrow=nrow, normalize=True, padding=1)
        #         plt.figure( figsize=(nrow,rows) )
        #         plt.imshow(grid.numpy().transpose((1, 2, 0)))
        #         plt.show()
                
        print('Backbone loaded')
    else:
        print('No backbone detected, training from scratch')
        contrastive_baselinemodel = Contrastive_Network(img_dim=data_settings['img_size'], num_classes=data_settings['num_output'], num_kernels=model_settings['num_kernels'],
            num_filters1=model_settings['num_filters1'], num_filters2=model_settings['num_filters2'], num_hidden=model_settings['num_hidden'],
            num_hidden2=model_settings['num_hidden2'], pooling_dim=model_settings['pooling_dim'], stride=model_settings['stride'], padding=model_settings['padding'],
            stridepool=model_settings['stridepool'], paddingpool=model_settings['paddingpool'], embedding_dim=model_settings['embedding_dim'])
    
    ckptfile = f"{cont_model_name}_ckpt.pth"
    contrastive_model = contrastive_baselinemodel.to(device)
    optimizer = torch.optim.Adam(list(contrastive_baselinemodel.parameters()), lr = train_settings['learning_rate'])
    loss_func = Contrastive_Loss(margin=train_settings['margin'])
    ckpt_epoch=0
    # Load checkpoint if possible
    if os.path.isfile(ckptfile):
        ckpt_epoch = load_checkpoint(contrastive_baselinemodel, optimizer, ckptfile)
        contrastive_model=contrastive_baselinemodel
        print('Checkpoint detected, starting from epoch {}'.format(ckpt_epoch))
    else:
        # contrastive_model = contrastive_baselinemodel.to(device)
        print('No checkpoint detected, starting from scratch')
    
    contrastive_model = contrastive_baselinemodel.to(device)
    # Initialise wandb logger
    # _____________________________________________________________ TURN OFF FOR DEBUGGING __________________________________________________________________________
    wandb_logger = Logger(f"inm705_Baseline_Contrastive_w_Backbone", project='inm705_CW')
    logger = wandb_logger.get_logger()
    # _____________________________________________________________ TURN OFF FOR DEBUGGING __________________________________________________________________________
    
    # variables for checkpoint saving
    max_test_acc = 0
    max_val_acc = 0
    min_loss = -1
    
    # _____________________________________________________________ TURN ON FOR DEBUGGING __________________________________________________________________________
    # plot_embeds(data_settings, contrastive_model, asl_train_testloader, epoch=ckpt_epoch, mode='Training', logger=None)
    # plot_embeds(data_settings, contrastive_model, asl_testloader, epoch=ckpt_epoch, mode='Testing', logger=None)
    # plot_embeds(data_settings, contrastive_model, asl_validloader, epoch=ckpt_epoch, mode='Validation', logger=None)
    # _____________________________________________________________ TURN ON FOR DEBUGGING __________________________________________________________________________
    # _____________________________________________________________ TURN OFF FOR DEBUGGING __________________________________________________________________________
    plot_embeds(data_settings, contrastive_model, asl_train_testloader, epoch=ckpt_epoch, mode='Training', logger=logger)
    plot_embeds(data_settings, contrastive_model, asl_testloader, epoch=ckpt_epoch, mode='Testing', logger=logger)
    plot_embeds(data_settings, contrastive_model, asl_validloader, epoch=ckpt_epoch, mode='Validation', logger=logger)
    # _____________________________________________________________ TURN OFF FOR DEBUGGING __________________________________________________________________________
    
    # Training loop per epoch
    for epoch in range(ckpt_epoch+1, train_settings['epochs']+1):
        total_loss = 0
        contrastive_model.train()
        
        for iter,(X1,X2,simi,y1) in enumerate(asl_trainloader):
            optimizer.zero_grad()
            X1, X2, simi = X1.to(device), X2.to(device), simi.to(device)
            ypred1, ypred2= contrastive_model(X1,X2)
            # print(ypred1.size())
            loss = loss_func(ypred1, ypred2, simi.long())

            # # print(loss)
            loss.backward()
            optimizer.step()
            total_loss+=loss
            
        print('Epoch:{}, Train Loss:{}'.format(epoch, total_loss/len(asl_trainloader)))
        
        # _____________________________________________________________ TURN ON FOR DEBUGGING __________________________________________________________________________
        # plot_embeds(data_settings, contrastive_model, asl_train_testloader, epoch=epoch, mode='Training', logger=None)
        # plot_embeds(data_settings, contrastive_model, asl_testloader, epoch=epoch, mode='Testing', logger=None)
        # plot_embeds(data_settings, contrastive_model, asl_validloader, epoch=epoch, mode='Validation', logger=None)
        
        # _____________________________________________________________ TURN OFF FOR DEBUGGING __________________________________________________________________________
        logger.log({'train_loss_Contrastive_No_Backbone': total_loss/len(asl_trainloader)})
        
        plot_embeds(data_settings, contrastive_model, asl_train_testloader, epoch=epoch, mode='Training', logger=logger)
        plot_embeds(data_settings, contrastive_model, asl_testloader, epoch=epoch, mode='Testing', logger=logger)
        plot_embeds(data_settings, contrastive_model, asl_validloader, epoch=epoch, mode='Validation', logger=logger)
        # _____________________________________________________________ TURN OFF FOR DEBUGGING __________________________________________________________________________
        
        if min_loss == -1:
            save_checkpoint(epoch, contrastive_model, cont_model_name, optimizer)
            min_loss = total_loss
        elif int(total_loss) < int(min_loss):
            save_checkpoint(epoch, contrastive_model, cont_model_name, optimizer)
            min_loss = total_loss

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