import os
import torch
import torch.nn as nn
import torchvision
from logger import Logger
from datasets import ASL_Dataset, ASL_Dataset_Contrastive
from models import BaselineCNN, Contrastive_Network, Contrastive_Loss, Contrastive_Embedder
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from utils import parse_arguments, read_settings, save_checkpoint, load_checkpoint, save_checkpoint_ZS
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

device = torch.device('cpu')
if torch.cuda.is_available():
   device = torch.device('cuda')
print('cuda' if torch.cuda.is_available() else 'cpu')

# Backbone name
model_name = 'BaselineCNN_ckpt'

# Contrastive model checkpoint name, change to "My_Contrastive_Backbone" and set config to 1 to swap models from CNCL to CNFCL, or give a random name to enable the model to start from scratch
cont_model_name = 'My_Contrastive_Backbone_Fixed'

def evaluate(data_settings, model, dataloader, mode='Training', logger=None):
    # Function for evaluating datasets
    # logger : the wandb logging class obtained from logger.py
    # data_settings properties are obtained from config.yaml
    # model : the model used for training
    # dataloader : the dataset to be evaluated in dataloader form
    # mode : Title to be logged on wandb
    
    
    # Preparations for evaluation
    model.eval()
    true_labels = []
    predicted_labels = []
    
    # Evaluate predictions with true outputs
    for X,y in dataloader:
        X, y = X.to(device), y.to(device)
        ypred = model(X)
        
        ypred = torch.argmax(ypred, axis=1, keepdims=False)

        # Convert to cpu variables to be translated to numpy
        true_labels.extend(y.cpu().detach().numpy())
        predicted_labels.extend(ypred.cpu().detach().numpy())

    # Calculate recall and precision
    overall_accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    
    print("Overall accuracy = {0:.4f}, precision = {1:.4f}, recall={2:.4f}".format(overall_accuracy, precision, recall))
    return overall_accuracy, precision

# Function for plotting embeddings
def plot_embeds_ZS(train_settings, model, dataloader, epoch, mode='Training', logger=None, ):
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
        
        cmap = plt.cm.get_cmap('hsv', 6)
        to_plot_pca = pca.fit_transform(embeddings)
        temp_labels=0
        plotted_labels = {}
        
        # Seperate Zero-shot labels from the original 26 datapoints
        for iter, label in enumerate(np.unique(labels)):
        # Create a dictionary to keep track of plotted labels
            class_indices = np.where(labels == label)[0]
            if label < 26:
                label_to_plot = 0
            else:
                if label not in plotted_labels:
                    # Add and plot label classes not in the original 26 classes
                    if 'temp_labels' not in locals():
                        temp_labels = 0
                    temp_labels += 1
                    label_to_plot = temp_labels
                    plotted_labels[label] = label_to_plot
                else:
                    label_to_plot = plotted_labels[label]

            # Plot with respect to PCA, set to 3 dimensions
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

        directory='inference_ZS'
        if not os.path.exists(directory):
            os.makedirs(directory)
        ax.legend()
        plt.savefig(os.path.join(directory,f"{mode}.png"), bbox_inches='tight')
        plt.close()
    return

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

        directory='inference_Contrastive'
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        ax.legend()
        plt.savefig(os.path.join(directory,f"{mode}_{epoch}.png"), bbox_inches='tight')
        plt.close()

def train(data_settings, model_settings, train_settings):
    
    # asl_dataset: Dataset with 87k datapoints and 29 classes
    asl_dataset = ASL_Dataset(mode='train', img_size=data_settings['img_size'], include_others=True)
    
    # Split datapoints
    data_len = len(asl_dataset)
    train_len = int(data_len*data_settings['train_size'])
    test_len = int((data_len - train_len)/2)
    val_len = data_len - train_len - test_len
    asl_dataset_train, asl_dataset_test, asl_dataset_valid = random_split(asl_dataset, [train_len, test_len, val_len])
    
    # Load data to dataloader
    asl_trainloader = DataLoader(asl_dataset_train, batch_size=data_settings['batch_size'], shuffle=True)
    asl_testloader = DataLoader(asl_dataset_test, batch_size=1, shuffle=False)
    asl_validloader = DataLoader(asl_dataset_valid, batch_size=1, shuffle=False)
    
    # asl_dataset_cont: original dataset with 87k datapoints, cut to 26 classes and 78k datapoints
    include_others = True
    asl_dataset_cont = ASL_Dataset_Contrastive(mode='train', img_size=data_settings['img_size'], include_others=include_others, simi_ratio=data_settings['simi_ratio'])
    
    # Split datapoints
    data_len_cont = len(asl_dataset_cont)
    train_len_cont = int(data_len_cont*data_settings['train_size'])
    test_len_cont = int((data_len_cont - train_len_cont)/2)
    val_len_cont = data_len_cont - train_len_cont - test_len_cont
    asl_dataset_train_cont, asl_dataset_test_cont, asl_dataset_valid_cont = random_split(asl_dataset_cont, [train_len_cont, test_len_cont, val_len_cont])
    
    # Load to dataloader
    asl_train_testloader_cont = DataLoader(asl_dataset_train_cont, batch_size=1, shuffle=False)
    asl_testloader_cont = DataLoader(asl_dataset_test_cont, batch_size=1, shuffle=False)
    asl_validloader_cont = DataLoader(asl_dataset_valid_cont, batch_size=1, shuffle=False)
    
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
        contrastive_baselinemodel = Contrastive_Network(backbone=backbone, type=model_settings['type'], img_dim=data_settings['img_size'], num_classes=data_settings['num_output'], num_kernels=model_settings['num_kernels'],
                                num_filters1=model_settings['num_filters1'], num_filters2=model_settings['num_filters2'], num_hidden=model_settings['num_hidden'],
                                num_hidden2=model_settings['num_hidden2'], pooling_dim=model_settings['pooling_dim'], stride=model_settings['stride'], padding=model_settings['padding'],
                                stridepool=model_settings['stridepool'], paddingpool=model_settings['paddingpool'])
                
        print('Backbone loaded')
    else:
        print('No Backbone detected, starting from scratch')
        contrastive_baselinemodel = Contrastive_Network(img_dim=data_settings['img_size'], num_classes=data_settings['num_output'], num_kernels=model_settings['num_kernels'],
                                num_filters1=model_settings['num_filters1'], num_filters2=model_settings['num_filters2'], num_hidden=model_settings['num_hidden'],
                                num_hidden2=model_settings['num_hidden2'], pooling_dim=model_settings['pooling_dim'], stride=model_settings['stride'], padding=model_settings['padding'],
                                stridepool=model_settings['stridepool'], paddingpool=model_settings['paddingpool'])
    
    ckptfile = f"{cont_model_name}_ckpt.pth"
    contrastive_model = contrastive_baselinemodel.to(device)
    optimizer = torch.optim.Adam(list(contrastive_baselinemodel.parameters()), lr = train_settings['learning_rate'])
    ckpt_epoch=0
    
    # Load checkpoint if possible
    if os.path.isfile(ckptfile):
        ckpt_epoch = load_checkpoint(contrastive_baselinemodel, optimizer, ckptfile)
        contrastive_model=contrastive_baselinemodel
        print('Checkpoint detected, starting from epoch {}'.format(ckpt_epoch))
    else:
        # Shouldn't happen in inference file
        raise Exception('No checkpoint detected')
    
    contrastive_model = contrastive_baselinemodel.to(device)
    baselinemodel = baselinemodel.to(device)    
    
    # Inference test for Baseline CNN
    train_acc, train_prec = evaluate(data_settings,baselinemodel,asl_trainloader, mode='Training')
    test_acc, test_prec = evaluate(data_settings,baselinemodel,asl_testloader, mode='Testing')
    val_acc, val_prec = evaluate(data_settings,baselinemodel,asl_validloader, mode='Validation')
    
    print(f'Training Accuracy: {train_acc}, Train Precision: {train_prec}')
    print(f'Testing Accuracy: {train_acc}, Testing Precision: {train_prec}')
    print(f'Validation Accuracy: {train_acc}, Validation Precision: {train_prec}')

    # Inference test for Zero Shot Learning
    plot_embeds_ZS(train_settings, contrastive_model, asl_train_testloader_cont, epoch=ckpt_epoch, mode='Training')
    plot_embeds_ZS(train_settings, contrastive_model, asl_testloader_cont, epoch=ckpt_epoch, mode='Testing')
    plot_embeds_ZS(train_settings, contrastive_model, asl_validloader_cont, epoch=ckpt_epoch, mode='Validation')
    
    # Inference test for Contrastive Learning Test
    plot_embeds(train_settings, contrastive_model, asl_train_testloader_cont, epoch=ckpt_epoch, mode='Training')
    plot_embeds(train_settings, contrastive_model, asl_testloader_cont, epoch=ckpt_epoch, mode='Testing')
    plot_embeds(train_settings, contrastive_model, asl_validloader_cont, epoch=ckpt_epoch, mode='Validation')

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