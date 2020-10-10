import os
import glob
import torch
import numpy                    as     np
import torch.nn                 as     nn
from   torch.utils.data         import Dataset, DataLoader
from   torch.utils.data.sampler import SubsetRandomSampler
from   utils_transform          import *


class aug_test():
    
    def __init__(self,model,h_flip,rot,device):
        self.model      = model
        self.horizontal = h_flip
        self.rotate     = rot
        self.device     = device
        self.sftmx      = nn.Softmax2d()

    def __call__(self,patch):
        aug_ls     = []
        img        = patch 
        tensor_img = torch.tensor(img)
        tensor_img = tensor_img.unsqueeze(0)
        aug_ls.append(tensor_img)
        img90      = self.rotate(img)
        tensor_img = torch.tensor(img90)
        tensor_img = tensor_img.unsqueeze(0)
        aug_ls.append(tensor_img)
        img180     = self.rotate(img90)
        tensor_img = torch.tensor(img180)
        tensor_img = tensor_img.unsqueeze(0)
        aug_ls.append(tensor_img)
        img270     = self.rotate(img180)
        tensor_img = torch.tensor(img270)
        tensor_img = tensor_img.unsqueeze(0)
        aug_ls.append(tensor_img)
        img_h      = self.horizontal(img)
        tensor_img = torch.tensor(img_h)
        tensor_img = tensor_img.unsqueeze(0)         
        aug_ls.append(tensor_img)
        img_h90    = self.rotate(img_h)
        tensor_img = torch.tensor(img_h90)
        tensor_img = tensor_img.unsqueeze(0)
        aug_ls.append(tensor_img)
        img_h180   = self.rotate(img_h90)
        tensor_img = torch.tensor(img_h180)
        tensor_img = tensor_img.unsqueeze(0)
        aug_ls.append(tensor_img)
        img_h270   = self.rotate(img_h180)
        tensor_img = torch.tensor(img_h270)
        tensor_img = tensor_img.unsqueeze(0)
        aug_ls.append(tensor_img)
        
        for index,img in enumerate(aug_ls):
            img        = img.float().to(self.device)
            output     = self.sftmx(self.model(img).cpu()).data.numpy()
            prediction = output[:,1,:,:]
            k          = -index%4
            rot        = rot_aug(k)
            if index == 0:
                x0   = prediction
            if index == 1:
                x1   = rot(prediction)
            if index == 2:
                x2   = rot(prediction)
            if index == 3:
                x3   = rot(prediction)
            if index == 4:
                x4   = self.horizontal(prediction)
            if index == 5:
                x5   = rot(prediction)
                x5   = self.horizontal(x5)
            if index == 6:
                x6   = rot(prediction)
                x6   = self.horizontal(x6)
            if index == 7:
                x7   = rot(prediction)
                x7   = self.horizontal(x7)
          
        combine = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7
        avg_combine = combine / len(aug_ls)
        return avg_combine    

    
class slide_func():
    
    def __init__(self,x0,y0,patchsize,inputsize,slidesize):
        self.patchsize = patchsize
        self.inputsize = inputsize
        self.slidesize = slidesize
        self.c         = []
        while y0 != None and x0 != None:
            x1,y1 = x0,y0
            self.c.append([x1,y1])
            x0,y0 = self.next_c(x1,y1)
            
    def __getitem__(self,idx):
        x,y = self.c[idx]
        return x,y
    
    def next_c(self,x,y):
        if x + self.patchsize == self.inputsize[1]:
            x = 0
        else:
            x += self.slidesize
            
        if x + self.patchsize > self.inputsize[1]:
            x = self.inputsize[1] - self.patchsize
            
        if x == 0:
            if y + self.patchsize == self.inputsize[0]:
                x = None
                y = None
            else:
                y +=self.slidesize
                if y + self.patchsize > self.inputsize[0]:
                    y = self.inputsize[0] - self.patchsize
        return x,y

    
def get_dataset(data_dir):
    data_ls   = glob.glob(os.path.join(data_dir,'n_train','*.npy'))#[:4000]
    target_ls = glob.glob(os.path.join(data_dir,'n_label','*.npy'))#[:4000]
#    seed1 =20
#    np.random.seed(seed1)
#    np.random.shuffle(data_ls)
#    np.random.seed(seed1)
#    np.random.shuffle(target_ls)
    full_data_ls   = []
    full_target_ls = []
    
    for index, img in enumerate(data_ls):
        data_array   = np.load(img)
        full_data_ls.append(data_array)
    for index,label in enumerate(target_ls):
        target_array = np.load(label)
        full_target_ls.append(target_array)
    
    return full_data_ls, full_target_ls
    

class convert(Dataset):
    
    def __init__(self,data_dir, transform=None):
        self.data_dir                = data_dir
        full_data_ls, full_target_ls = get_dataset(self.data_dir)
        self.data_ls                 = full_data_ls
        self.target_ls               = full_target_ls
        self.transform               = transform
        
    def __len__(self):
        return len(self.data_ls)
        
    def __getitem__(self, index):
        img        = self.data_ls[index]
        img        = img.transpose(2,0,1)
        target     = self.target_ls[index]
        img,target = self.transform(img),self.transform(target)
        return img, target


def get_loader(data_dir,
               batch_size,
               patchsize,
               seed        = 10,
               val_split   = 0.2,
               num_workers = 4):
    
    train_transform = compose_transforms([
        randcrop(patchsize),
        hor_flip(),
        vert_flip(),
        rot(),
        brightness(),
        norm(),
        to_tensor()
    ])
    
    val_transform  = compose_transforms([
        centercrop(patchsize),
        norm(),
        to_tensor()
    ])
    
    train_dataset = convert(data_dir=data_dir, transform=train_transform)
    val_dataset   = convert(data_dir=data_dir, transform=val_transform)
    datasize      = len(train_dataset)
    indicies      = list(range(datasize))
    split         = int(np.floor(val_split * datasize))

    np.random.seed(seed)
    np.random.shuffle(indicies)
    
    train_indicies, val_indicies = indicies[split:], indicies[:split]
    train_sampler                = SubsetRandomSampler(train_indicies)
    val_sampler                  = SubsetRandomSampler(val_indicies)
    
    train_loader=DataLoader(train_dataset,
                            batch_size  = batch_size,
                            sampler     = train_sampler,
                            num_workers = num_workers
                           )
    val_loader=DataLoader(val_dataset,
                            batch_size  = batch_size,
                            sampler     = val_sampler,
                            num_workers = num_workers
                           )

    return train_loader,val_loader