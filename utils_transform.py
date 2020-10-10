import numpy as np
import random
import torch
import torch.nn.functional as F
import pdb
import cv2


class compose_transforms:
    def __init__(self, transform):
        self.transform = transform
    def __call__(self,array):
        for t in self.transform:
            array=t(array)
        return array
    def __repr__(self):
        list_transform=[str(transform) for transform in self.transform]
        list_transform= '\n'.join(list_transform)
        return list_transform


class centercrop():
    def __init__(self,patchsize):
        self.patchsize=patchsize
    def __call__(self,array):
        if array.ndim==3:
            _,y,x=array.shape
            starty=y//2-(self.patchsize//2)
            startx=x//2-(self.patchsize//2)
            return array[:, starty:starty+self.patchsize, startx:startx+self.patchsize].copy()
        elif array.ndim==2:
            y,x=array.shape
            starty=y//2-(self.patchsize//2)
            startx=x//2-(self.patchsize//2)
            return array[starty:starty+self.patchsize, startx:startx+self.patchsize].copy()
    
    
class randcrop():
    def __init__(self,patchsize):
        self.patchsize=patchsize
        self.n_h=np.random.randint(0, 256 - self.patchsize)
        self.n_w=np.random.randint(0, 256 - self.patchsize)
    def __call__(self,array):
        if array.ndim==3:
            return array[:, self.n_h : self.n_h + self.patchsize,
                         self.n_w : self.n_w + self.patchsize].copy()
        elif array.ndim==2:
            return array[self.n_h : self.n_h + self.patchsize,
                         self.n_w : self.n_w + self.patchsize].copy()


class hor_flip():
    def __init__(self,probability=0.5):
        self.probability=probability
        self.n=np.random.rand()
    def __call__(self,array):
        if self.n < self.probability:
            if array.ndim==3:
                return array[:,:,::-1].copy()
            elif array.ndim==2:
                return array[:,::-1].copy()
        else:
            return array

        
class vert_flip():
    def __init__(self,probability=0.5):
        self.probability=probability
        self.n=np.random.rand()
    def __call__(self,array):
        if self.n < self.probability:
            if array.ndim==3:
                return array[:,::-1,:].copy()
            elif array.ndim==2:
                return array[::-1,:].copy()
        else:
            return array


class norm():
    def __call__(self,array):
        if array.ndim==3:
            array=array/255

            # for train_try
            #mean=np.array([[[114.32559481]], [[119.42060747]], [[116.79047923]]])
            #std=np.array([[[44.9699212]], [[41.11387233]], [[37.83859202]]])

            # for train_200per
            #mean=np.array([[[115.05077418]], [[120.2706386]], [[117.55850015]]])
            #std=np.array([[[44.91689441]], [[41.08925198]], [[37.73248225]]])

            #array=(array-mean)/std
            return array
        if array.ndim==2:
            return array

        
class to_tensor():
    def __call__(self,array):
        if array.ndim==3:
            return torch.from_numpy(array)
        if array.ndim==2:
            array = torch.from_numpy(array.astype(int))
            return array
       
    
class brightness():
    def __call__(self,array):
        if array.ndim==3:
            img_HLS=cv2.cvtColor(array.transpose(1,2,0),cv2.COLOR_RGB2HLS)
            img_HLS=np.array(img_HLS, dtype=np.float64)
            bright_coeff=np.random.uniform()+0.5
            img_HLS[:,:,1]=img_HLS[:,:,1]*bright_coeff
            img_HLS[:,:,1][img_HLS[:,:,1]>255]=255
            img_HLS=np.array(img_HLS, dtype=np.uint8)
            img_RGB=cv2.cvtColor(img_HLS,cv2.COLOR_HLS2RGB).transpose(2,0,1)
            return img_RGB
        if array.ndim==2:
            return array
        

class rot():
    def __init__(self,probability=0.5):
        self.probability=probability
        self.n=np.random.rand()
        self.k=np.random.randint(1,4)
    def __call__(self,array):
        if self.n < self.probability:
            if array.ndim==3:
                return  np.rot90(array,k=self.k,axes=(1,2)).copy()
            elif array.ndim==2:
                return np.rot90(array,k=self.k).copy()
        else:
            return array
     
    
class rot_aug():
    def __init__(self, k):
        self.k=k
    def __call__(self,array):
        return  np.rot90(array,k=self.k,axes=(1,2)).copy()
            


