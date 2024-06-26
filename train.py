import os
import torch
from logger import Logger
from datasets import ASL_Dataset
from models import BaselineCNN
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from utils import parse_arguments, read_settings, save_checkpoint, load_checkpoint

device = torch.device('cpu')
if torch.cuda.is_available():
   device = torch.device('cuda')
print('cuda' if torch.cuda.is_available() else 'cpu')

model_name = 'BaselineCNN'

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
    
    logger.log({f"{mode} Precision": precision,
                f"{mode} Recall": recall,
    })
    
    print("Overall accuracy = {0:.4f}, precision = {1:.4f}, recall={2:.4f}".format(overall_accuracy, precision, recall))
    return overall_accuracy, precision

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

    
    # Baseline model for evaluating
    baselinemodel = BaselineCNN(img_dim=data_settings['img_size'], num_classes=data_settings['num_output'], num_kernels=model_settings['num_kernels'],
                                num_filters1=model_settings['num_filters1'], num_filters2=model_settings['num_filters2'], num_hidden=model_settings['num_hidden'],
                                num_hidden2=model_settings['num_hidden2'], pooling_dim=model_settings['pooling_dim'], stride=model_settings['stride'], padding=model_settings['padding'],
                                stridepool=model_settings['stridepool'], paddingpool=model_settings['paddingpool'])
    
    # Define training parameters
    ckptfile = f"{model_name}_ckpt.pth"
    baselinemodel = baselinemodel.to(device)   
    optimizer = torch.optim.Adam(list(baselinemodel.parameters()), lr = train_settings['learning_rate_backbone'])
    
    # Load checkpoint if possible
    if os.path.isfile(ckptfile):
        print('Checkpoint deteced')
        load_checkpoint(baselinemodel, optimizer, ckptfile)
        print('checkpoint loaded')
    else:
        print('Starting from scratch')
        
    baselinemodel = baselinemodel.to(device)    
    
    
    wandb_logger = Logger(f"inm705_Backbone_1", project='inm705_CW')
    logger = wandb_logger.get_logger()
    
    # variables for checkpoint saving
    max_test_acc = 0
    max_val_acc = 0
    
    train_acc, train_prec = evaluate(data_settings,baselinemodel,asl_trainloader, mode='Training', logger=logger)
    test_acc, test_prec = evaluate(data_settings,baselinemodel,asl_testloader, mode='Testing', logger=logger)
    val_acc, val_prec = evaluate(data_settings,baselinemodel,asl_validloader, mode='Validation', logger=logger)

    
    # Training loop per epoch
    # for epoch in range(train_settings['epochs_backbone']):
    #     total_loss = 0
    #     baselinemodel.train()
        
    #     for iter,(X,y) in enumerate(asl_trainloader):
    #         optimizer.zero_grad()
    #         X, y = X.to(device), y.to(device)
    #         ypred = baselinemodel(X)
    #         loss = F.cross_entropy(ypred, y)
    #         # print(loss)
    #         loss.backward()
    #         optimizer.step()
    #         total_loss+=loss
            
    #     logger.log({'train_loss': total_loss/len(asl_trainloader)})
    #     print('Epoch:{}, Train Loss:{}'.format(epoch, total_loss/len(asl_trainloader)))
        
    #     train_acc, train_prec = evaluate(data_settings,baselinemodel,asl_trainloader, mode='Training', logger=logger)
    #     test_acc, test_prec = evaluate(data_settings,baselinemodel,asl_testloader, mode='Testing', logger=logger)
    #     val_acc, val_prec = evaluate(data_settings,baselinemodel,asl_validloader, mode='Validation', logger=logger)

    #     # Save best model during training time based on testing and validation accuracies
    #     if((test_acc > max_test_acc) and (val_acc > max_val_acc)):
    #         save_checkpoint(epoch, baselinemodel, f'{model_name}_{epoch}', optimizer)
    #         max_test_acc = test_acc
    #         max_val_acc = val_acc

    # return

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