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

# os.environ['https_proxy'] = "http://hpc-proxy00.city.ac.uk:3128"398
device = torch.device('cpu')
if torch.cuda.is_available():
   device = torch.device('cuda')
print('cuda' if torch.cuda.is_available() else 'cpu')

model_name = 'BaselineCNN_ckpt'
cont_model_name = 'My_Contrastive_ZS'

def plot_embeds(data_settings, model, dataloader, epoch, mode='Training', logger=None, ):
    # Preparations for evaluation
    model.eval()
    embeddings=[]
    labels=[]
    tsne = TSNE(n_components=3, random_state=42)
    pca = PCA(n_components=3, random_state=42)
    fig=plt.figure(figsize=(8,6))
    ax = fig.add_subplot(projection='3d')
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
        cmap = plt.cm.get_cmap('hsv', 6)
        # to_plot = tsne.fit_transform(embeddings)
        to_plot_pca = pca.fit_transform(embeddings)
        temp_labels=0
        plotted_labels = {}
        for iter, label in enumerate(np.unique(labels)):
        # Create a dictionary to keep track of plotted labels

            class_indices = np.where(labels == label)[0]
            if label < 26:
                label_to_plot = 0
            else:
                if label not in plotted_labels:
                    # Increment label_to_plot only if the label is not plotted before
                    if 'temp_labels' not in locals():
                        temp_labels = 0
                    temp_labels += 1
                    label_to_plot = temp_labels
                    plotted_labels[label] = label_to_plot
                else:
                    # If label is already plotted, get its assigned label_to_plot
                    label_to_plot = plotted_labels[label]

            if iter == 0:
                ax.scatter(to_plot_pca[class_indices, 0], to_plot_pca[class_indices, 1], zs=to_plot_pca[class_indices, 2],
                        zdir='z', label=label_to_plot, c=[cmap(label_to_plot)])
            else:
                ax.scatter(to_plot_pca[class_indices, 0], to_plot_pca[class_indices, 1], zs=to_plot_pca[class_indices, 2],
                        zdir='z', c=[cmap(label_to_plot)])

        # Add legend entries for unique labels
        for label, label_to_plot in plotted_labels.items():
            ax.scatter([], [], [], label=label_to_plot, c=[cmap(label_to_plot)])

        ax.legend()
            # ax.scatter(to_plot_pca[class_indices,0], to_plot_pca[class_indices,1], label=label_to_plot, c=[cmap(label_to_plot)])
        # ax.scatter(to_plot_pca[:,0], to_plot_pca[:,1], zs=to_plot_pca[:,2], zdir='z', c=labels[:])
        # plt.show()
        directory='plot_images_5'
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
    include_others = True
    asl_dataset = ASL_Dataset_Contrastive(mode='test', img_size=data_settings['img_size'], include_others=include_others, simi_ratio=data_settings['simi_ratio'])
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
        # print(list(backbone.children())[:4])
        backbone.load_state_dict(backbone_ckpt['model_state'])
        contrastive_baselinemodel = Contrastive_Network(backbone=backbone, img_dim=data_settings['img_size'], num_classes=data_settings['num_output'], num_kernels=model_settings['num_kernels'],
                                num_filters1=model_settings['num_filters1'], num_filters2=model_settings['num_filters2'], num_hidden=model_settings['num_hidden'],
                                num_hidden2=model_settings['num_hidden2'], pooling_dim=model_settings['pooling_dim'], stride=model_settings['stride'], padding=model_settings['padding'],
                                stridepool=model_settings['stridepool'], paddingpool=model_settings['paddingpool'])
                
        print('Backbone loaded')
    else:
        raise Exception('Error: No Backbone Detected')
    
    ckptfile = f"{cont_model_name}_ckpt.pth"
    contrastive_model = contrastive_baselinemodel.to(device)
    print(contrastive_model.parameters())
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
    # wandb_logger = Logger(f"inm705_Zero_Shot", project='inm705_CW')
    # logger = wandb_logger.get_logger()
    # _____________________________________________________________ TURN OFF FOR DEBUGGING __________________________________________________________________________
    
    # variables for checkpoint saving
    max_test_acc = 0
    max_val_acc = 0
    epoch = 1
    # plot_embeds(data_settings, contrastive_model, asl_train_testloader, epoch=ckpt_epoch, mode='Training', logger=logger)
    # plot_embeds(data_settings, contrastive_model, asl_testloader, epoch=ckpt_epoch, mode='Testing', logger=logger)
    # plot_embeds(data_settings, contrastive_model, asl_validloader, epoch=ckpt_epoch, mode='Validation', logger=logger)
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
        plot_embeds(data_settings, contrastive_model, asl_train_testloader, epoch=epoch, mode='Training', logger=None)
        plot_embeds(data_settings, contrastive_model, asl_testloader, epoch=epoch, mode='Testing', logger=None)
        plot_embeds(data_settings, contrastive_model, asl_validloader, epoch=epoch, mode='Validation', logger=None)
        
        # _____________________________________________________________ TURN OFF FOR DEBUGGING __________________________________________________________________________
        # logger.log({'train_loss': total_loss/len(asl_trainloader)})
        
        # plot_embeds(data_settings, contrastive_model, asl_train_testloader, epoch=epoch, mode='Training', logger=logger)
        # plot_embeds(data_settings, contrastive_model, asl_testloader, epoch=epoch, mode='Testing', logger=logger)
        # plot_embeds(data_settings, contrastive_model, asl_validloader, epoch=epoch, mode='Validation', logger=logger)
        # _____________________________________________________________ TURN OFF FOR DEBUGGING __________________________________________________________________________
        
        
        # train_acc, train_prec = evaluate(data_settings,baselinemodel,asl_trainloader, mode='Training')
        # test_acc, test_prec = evaluate(data_settings,baselinemodel,asl_testloader, mode='Testing')
        # val_acc, val_prec = evaluate(data_settings,baselinemodel,asl_validloader, mode='Validation')

        
    save_checkpoint(epoch, contrastive_model, cont_model_name, optimizer)

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