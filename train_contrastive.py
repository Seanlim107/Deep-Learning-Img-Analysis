import os
import torch
from logger import Logger
from datasets import ASL_Dataset_Contrastive
from models import BaselineCNN, Contrastive_Network, Contrastive_Loss
from torch.utils.data import DataLoader, random_split
from utils import parse_arguments, read_settings, save_checkpoint, load_checkpoint
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


device = torch.device('cpu')
if torch.cuda.is_available():
   device = torch.device('cuda')
print('cuda' if torch.cuda.is_available() else 'cpu')

# Backbone name
model_name = 'BaselineCNN_ckpt2'

# Contrastive model checkpoint name, change to "My_Contrastive_Backbone" and set config to 1 to swap models from CNCL to CNFCL, or give a random name to enable the model to start from scratch
cont_model_name = 'My_Contrastive_No_Backbone'

# Function for plotting embeddings
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
        labels=np.array(labels)
        embeddings=np.concatenate(embeddings, axis=0)
        
        # Plot for 26 classes
        cmap = plt.cm.get_cmap('tab20', len(np.unique(labels)))
        to_plot_pca = pca.fit_transform(embeddings)
        for label in np.unique(labels):
            class_indices=np.where(labels==label)[0]
            ax.scatter(to_plot_pca[class_indices,0], to_plot_pca[class_indices,1], zs=to_plot_pca[class_indices,2], zdir='z', label=label, c=[cmap(label)])
        
        directory='plot_images_Baseline'
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        ax.legend()
        plt.savefig(os.path.join(directory,f"{mode}_{epoch}.png"), bbox_inches='tight')
        plt.close()
    return

def train(data_settings, model_settings, train_settings):
    
    # asl_dataset: original dataset with 87k datapoints, cut to 26 classes and 78k datapoints
    include_others = False
    asl_dataset = ASL_Dataset_Contrastive(mode='train', img_size=data_settings['img_size'], simi_ratio=data_settings['simi_ratio'], include_others=include_others)
    
    # Split datapoints
    data_len = len(asl_dataset)
    train_len = int(data_len*data_settings['train_size'])
    test_len = int((data_len - train_len)/2)
    val_len = data_len - train_len - test_len
    asl_dataset_train, asl_dataset_test, asl_dataset_valid = random_split(asl_dataset, [train_len, test_len, val_len])
    
    # Load data to dataloader
    asl_trainloader = DataLoader(asl_dataset_train, batch_size=data_settings['batch_size'], shuffle=True)
    asl_train_testloader = DataLoader(asl_dataset_train, batch_size=1, shuffle=False)
    asl_testloader = DataLoader(asl_dataset_test, batch_size=1, shuffle=False)
    asl_validloader = DataLoader(asl_dataset_valid, batch_size=1, shuffle=False)
    
    # Baseline model for evaluating
    baselinemodel = BaselineCNN(img_dim=data_settings['img_size'], num_classes=data_settings['num_output'], num_kernels=model_settings['num_kernels'],
                                num_filters1=model_settings['num_filters1'], num_filters2=model_settings['num_filters2'], num_hidden=model_settings['num_hidden'],
                                num_hidden2=model_settings['num_hidden2'], pooling_dim=model_settings['pooling_dim'], stride=model_settings['stride'], padding=model_settings['padding'],
                                stridepool=model_settings['stridepool'], paddingpool=model_settings['paddingpool'])
     
    backbonefile = f"{model_name}.pth"
    
    # Load backbone if exists else start from scratch
    if os.path.isfile(backbonefile):
        print('Backbone detected, installing backbone')
        backbone_ckpt = torch.load(backbonefile, map_location=device)
        backbone=baselinemodel
        
        backbone.load_state_dict(backbone_ckpt['model_state'])
        contrastive_baselinemodel = Contrastive_Network(backbone=backbone, embedding_dim=model_settings['embedding_dim'])
                
        print('Backbone loaded')
    else:
        print('No backbone detected, training from scratch')
        contrastive_baselinemodel = Contrastive_Network(img_dim=data_settings['img_size'], type=model_settings['type'], num_classes=data_settings['num_output'], num_kernels=model_settings['num_kernels'],
            num_filters1=model_settings['num_filters1'], num_filters2=model_settings['num_filters2'], num_hidden=model_settings['num_hidden'],
            num_hidden2=model_settings['num_hidden2'], pooling_dim=model_settings['pooling_dim'], stride=model_settings['stride'], padding=model_settings['padding'],
            stridepool=model_settings['stridepool'], paddingpool=model_settings['paddingpool'], embedding_dim=model_settings['embedding_dim'])
    
    # Define training parameters
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
        print('No checkpoint detected, starting from scratch')
    
    contrastive_model = contrastive_baselinemodel.to(device)
    
    # Initialise wandb logger
    wandb_logger = Logger(f"inm705_Baseline_Contrastive_w_Backbone", project='inm705_CW')
    logger = wandb_logger.get_logger()
    
    # variables for checkpoint saving
    min_loss = -1
    
    # Plotting for initialization stage, before training
    plot_embeds(data_settings, contrastive_model, asl_train_testloader, epoch=ckpt_epoch, mode='Training', logger=logger)
    plot_embeds(data_settings, contrastive_model, asl_testloader, epoch=ckpt_epoch, mode='Testing', logger=logger)
    plot_embeds(data_settings, contrastive_model, asl_validloader, epoch=ckpt_epoch, mode='Validation', logger=logger)
    
    # Training loop per epoch
    for epoch in range(ckpt_epoch+1, train_settings['epochs']+1):
        total_loss = 0
        contrastive_model.train()
        
        for iter,(X1,X2,simi,y1) in enumerate(asl_trainloader):
            optimizer.zero_grad()
            X1, X2, simi = X1.to(device), X2.to(device), simi.to(device)
            ypred1, ypred2= contrastive_model(X1,X2)
            loss = loss_func(ypred1, ypred2, simi.long())

            loss.backward()
            optimizer.step()
            total_loss+=loss
            
        print('Epoch:{}, Train Loss:{}'.format(epoch, total_loss/len(asl_trainloader)))
        
        logger.log({'train_loss_Contrastive_No_Backbone': total_loss/len(asl_trainloader)})
        
        plot_embeds(data_settings, contrastive_model, asl_train_testloader, epoch=epoch, mode='Training', logger=logger)
        plot_embeds(data_settings, contrastive_model, asl_testloader, epoch=epoch, mode='Testing', logger=logger)
        plot_embeds(data_settings, contrastive_model, asl_validloader, epoch=epoch, mode='Validation', logger=logger)
        
        # Save Checkpoints        
        if min_loss == -1:
            save_checkpoint(epoch, contrastive_model, cont_model_name, optimizer)
            min_loss = total_loss
        elif int(total_loss) < int(min_loss):
            save_checkpoint(epoch, contrastive_model, cont_model_name, optimizer)
            min_loss = total_loss

    return

def main():
    args = parse_arguments()

    # Read settings from the YAML file
    settings = read_settings(args.config)

    # Access and use the settings as needed
    data_settings = settings.get('data', {})
    model_settings = settings.get('model', {})
    train_settings = settings.get('train', {})

    train(data_settings, model_settings, train_settings)
    
    
if __name__ == '__main__':
    main()