import os
import pdb
import glob
import time
import torch
import argparse
import numpy          as     np
import pandas         as     pd
import torch.nn       as     nn
import torch.optim    as     optim
from   tqdm           import tqdm
from   model          import NestedUNet
from   utils          import get_loader
from   NestedResUNet  import NestedResUNet


parser = argparse.ArgumentParser(description='building detection')
parser.add_argument('exp_name',                                   help='name of this experiment(output folder)')
parser.add_argument('--data_dir',  type=str,   default='./data/', help='directory that contains training data')
parser.add_argument('--nEpochs',   type=int,   default=100,       help='number of epochs to train for')
parser.add_argument('--patchsize', type=int,   default=224,       help='training patch size')
parser.add_argument('--batchsize', type=int,   default=4,         help='training/test batch size')
parser.add_argument('--lr',        type=float, default=0.003,     help='learning rate. Default=0.003')
parser.add_argument('--wd',        type=float, default=0.001,     help='L2-weight decay. Default=0.001')
args   = parser.parse_args()


device = torch.device('cuda')

#Dataloaders
train_loader, val_loader = get_loader(args.data_dir, args.batchsize, args.patchsize)

#Model

model     = NestedResUNet().to(device)
#model     = NestedUNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10, verbose=True, min_lr=0.0001)

#print(model)


def train(epoch):
    model.train()
    sfmx           = nn.Softmax2d()
    avg_train_loss = 0
    TP             = 0
    TN             = 0
    FP             = 0
    FN             = 0
    for iteration, batch in tqdm(enumerate(train_loader, 1)):
        input, target   = batch
        input, target   = input.float(), target.long()
        input, target   = input.to(device), target.to(device)
        prediction      = model(input)
        prediction      = sfmx(prediction)
        loss            = criterion(prediction, target)
        avg_train_loss += loss.item()
        _, predicted    = torch.max(prediction, 1)
        
        TP += (target + predicted  ==  2).sum().item()
        TN += (target + predicted  ==  0).sum().item()
        FP += (target - predicted  == -1).sum().item()
        FN += (target - predicted  ==  1).sum().item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    avg_train_loss  = avg_train_loss / len(train_loader)
    recall_train    = TP / (TP + FN + 1e-7)
    precision_train = TP / (TP + FP + 1e-7)
    accuracy_train  = (TP + TN) / (TP + TN + FP + FN + 1e-7)    
    
    print("---Epoch[{}]---\n===>  Avg. Train Loss: {:.4f}  Train Acc: {:.4f}  Train Recall: {:.4f}  Train Precision: {:.4f}".format(epoch, avg_train_loss, accuracy_train, recall_train, precision_train))
    return avg_train_loss, recall_train, precision_train, accuracy_train
        

def validate():
    model.eval()
    sfmx         = nn.Softmax2d()
    avg_val_loss = 0
    TP           = 0
    TN           = 0
    FP           = 0
    FN           = 0
    for batch in tqdm(val_loader):
        with torch.no_grad():
            input, target = batch
            input, target = input.float(), target.long()
            input, target = input.to(device), target.to(device)
            prediction    = model(input)
            prediction    = sfmx(prediction)
            loss          = criterion(prediction, target)
            avg_val_loss += loss.item()
            _, predicted  = torch.max(prediction, 1)
        
            TP += (target + predicted  ==  2).sum().item()
            TN += (target + predicted  ==  0).sum().item()
            FP += (target - predicted  == -1).sum().item()
            FN += (target - predicted  ==  1).sum().item()
    
    avg_val_loss  = avg_val_loss / len(val_loader)
    recall_val    = TP / (TP + FN + 1e-7)
    precision_val = TP / (TP + FP + 1e-7)
    accuracy_val  = (TP + TN) / (TP + TN + FP + FN + 1e-7)
    F1_val        = 2*precision_val*recall_val/(precision_val+recall_val+ 1e-7)


    print("===>  Avg. Val Loss: {:.4f}  Val Acc: {:.4f}  Val Recall: {:.4f}  Val Precision: {:.4f}  ***F1: {:.4f}***".format(avg_val_loss, accuracy_val, recall_val, precision_val,F1_val))
    return avg_val_loss, recall_val, precision_val, accuracy_val, F1_val
          
    
def save_model(epoch, loss):
    os.makedirs('saved_models', exist_ok=True)
    path   = 'saved_models/{}.pth'.format(args.exp_name)
    torch.save(model, path)
    print('Best model saved to {}\n'.format(path))
    
    with open(os.path.join('saved_models', 'log'), 'a') as f:
        f.write('{}[Epoch:{:>3}] loss: {:.4f}\n'.format(args.exp_name, epoch, loss))
      
    
min_loss = 100
Train_log  = pd.DataFrame(columns=['epoch','avg_train_loss', 'recall_train', 'precision_train', 'accuracy_train', 'avg_val_loss', 'recall_val', 'precision_val', 'accuracy_val', 'F1_val'])    

for epoch in range(1, args.nEpochs + 1):
    avg_train_loss, recall_train, precision_train, accuracy_train = train(epoch)
    avg_val_loss, recall_val, precision_val, accuracy_val, F1_val = validate()
    scheduler.step(avg_val_loss)
    log                                                           = pd.DataFrame([(epoch,avg_train_loss,recall_train,precision_train,accuracy_train,avg_val_loss,recall_val,precision_val,accuracy_val,F1_val)], columns=['epoch','avg_train_loss', 'recall_train', 'precision_train', 'accuracy_train', 'avg_val_loss', 'recall_val', 'precision_val','accuracy_val','F1_val'])
    Train_log                                                     = pd.concat([Train_log,log])
    Train_log.to_csv('./results/{}.csv'.format(args.exp_name), index=False)
    
    if avg_val_loss < min_loss:
        save_model(epoch, avg_val_loss)
        min_loss = avg_val_loss