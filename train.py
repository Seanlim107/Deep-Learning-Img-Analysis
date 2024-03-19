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
from utils import parse_arguments, read_settings

# os.environ['https_proxy'] = "http://hpc-proxy00.city.ac.uk:3128"
device = torch.device('cpu')
if torch.cuda.is_available():
   device = torch.device('cuda')
print('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(logger, data_settings, model, dataloader, mode='Training'):
    model.eval()
    # num_outputs = data_settings['num_output']
    true_labels = []
    predicted_labels = []
    
    for X,y in dataloader:
        X, y = X.to(device), y.to(device)
        ypred = model(X)
        
        ypred = torch.argmax(ypred, axis=1, keepdims=False)
        # print(ypred)
        
        true_labels.append(y.item())
        predicted_labels.append(ypred.item())
    # print(ypred, y)
    overall_accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    
    logger.log({"{} Overall Accuracy".format(mode): overall_accuracy,
                "{} Precision".format(mode): precision,
                "{} Recall".format(mode): recall,
    })
    print("Overall accuracy = {0:.4f}, precision = {1:.4f}, recall={2:.4f}".format(overall_accuracy, precision, recall))
    # print(true_labels[:32],predicted_labels[:32])
    return

def train(data_settings, model_settings, train_settings):
    
    asl_dataset = ASL_Dataset(mode='train', img_size=data_settings['img_size'])
    data_len = len(asl_dataset)
    train_len = int(data_len*data_settings['train_size'])
    test_len = int((data_len - train_len)/2)
    val_len = data_len - train_len - test_len
    asl_dataset_train, asl_dataset_test, asl_dataset_valid = random_split(asl_dataset, [train_len, test_len, val_len])
    
    asl_trainloader = DataLoader(asl_dataset_train, batch_size=data_settings['batch_size'], shuffle=True)
    asl_testloader = DataLoader(asl_dataset_test, batch_size=1, shuffle=False)
    asl_validloader = DataLoader(asl_dataset_valid, batch_size=1, shuffle=False)
    
    asl_bb_dataset_train = ASL_BB_Dataset(mode='train')
    asl_bb_testloader = DataLoader(asl_bb_dataset_train, batch_size=1, shuffle=False)
    
    baselinemodel = BaselineCNN(img_dim=data_settings['img_size'], num_classes=data_settings['num_output'], num_kernels=model_settings['num_kernels'],
                                num_filters1=model_settings['num_filters1'], num_filters2=model_settings['num_filters2'], num_hidden=model_settings['num_hidden'],
                                pooling_dim=model_settings['pooling_dim'], stride=model_settings['stride'], padding=model_settings['padding'],
                                stridepool=model_settings['stridepool'], paddingpool=model_settings['paddingpool'])
    
    baselinemodel.train()
    baselinemodel = baselinemodel.to(device)
    
    optimizer = torch.optim.Adam(list(baselinemodel.parameters()), lr = train_settings['learning_rate'])
    wandb_logger = Logger(
        f"inm705_Baseline CNN",
        project='inm705_CW')
    logger = wandb_logger.get_logger()
    
    for epoch in range(train_settings['epochs']):
        total_loss = 0
        baselinemodel.train()
        for iter,(X,y) in enumerate(asl_trainloader):
            # print(X)
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
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))
        evaluate(logger,data_settings,baselinemodel,asl_trainloader, mode='Training')
        evaluate(logger,data_settings,baselinemodel,asl_testloader, mode='Testing')
        evaluate(logger,data_settings,baselinemodel,asl_validloader, mode='Validation')
        evaluate(logger,data_settings,baselinemodel,asl_bb_testloader, mode='Testing BB')
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