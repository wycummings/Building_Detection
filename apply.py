import torch
import argparse
import glob
import os
import pdb
import numpy           as     np
import torch.nn        as     nn
import PIL.Image       as     Image
from   utils_transform import *
from   utils           import aug_test, slide_func
from   tqdm            import tqdm

parser = argparse.ArgumentParser(description='building detection')
parser.add_argument('trained_model',                                                               help='trained model')
parser.add_argument('--data_dir',  type=str,  default='/export_afenfs_large-capacity/shared/iis/.DEBUG/nojima/projects/intern_compe/building_detection/data_link/test/image/', help='directory that contains training data')
parser.add_argument('--patchsize', type=int,  default=448,                                         help='training patch size')
parser.add_argument('--slidesize', type=int,  default=448,                                         help='training patch size')
parser.add_argument('--filename',  type=str,  default='will_ResUNet224v6.txt',                    help='output file name')
parser.add_argument('--augument',  type=bool, default=False,                                        help='output file name')
args = parser.parse_args()

device = torch.device('cuda')


torch.nn.Module.dump_patches = True

model = torch.load(args.trained_model)
model.to(device)
model.eval()


def get_test_ls(data_dir):
    full_test_ls = []
    test_ls      = (glob.glob(os.path.join(data_dir, '*.tif')))
    test_ls.sort()
  
    
    for index, img in enumerate(test_ls):
        test_img   = Image.open(img)
        test_array = np.asarray(test_img)
        full_test_ls.append(test_array)
        
    return full_test_ls, test_ls

    
full_test_ls, test_ls = get_test_ls(args.data_dir)

if not os.path.exists('apply_results/224v6_{}'.format(args.trained_model[13:-4])):
    os.mkdir('apply_results/224v6_{}'.format(args.trained_model[13:-4]))
    
path      = 'apply_results/224v6_{}'.format(args.trained_model[13:-4])
n_path    = os.path.join('./'+path, args.filename)
wdr       = '/export_afenfs_large-capacity/shared/iis/.DEBUG/will/building_classification/'+path+'/'
f         = open(n_path, "w+")
j         = 0
normalize = norm()
h_flip    = hor_flip(probability=1)
rot       = rot_aug(k=1)
sftmx     = nn.Softmax2d()
augument  = aug_test(model, h_flip, rot, device)

for i in tqdm(full_test_ls):
    i  = normalize(i)
    x0 = 0
    y0 = 0
    output    = np.zeros([1,1,i.shape[0],i.shape[1]])
    layer1    = np.zeros([1,1,i.shape[0],i.shape[1]])
    file_name = test_ls[j]
    file_name = os.path.basename(file_name)
    j += 1
    
    for x,y in slide_func(x0,y0,args.patchsize,i.shape,args.slidesize):
        patch = i[y:y+args.patchsize,x:x+args.patchsize,:]
        patch = np.transpose(patch,(2,0,1))
        
        if args.augument == True:
            prediction = augument(patch)
            
        else:
            patch      = torch.tensor(patch)
            patch      = patch.unsqueeze(0)
            patch      = patch.float().to(device)
            pred       = sftmx(model(patch).cpu()).data.numpy()
            prediction = pred[:,1,:,:]
            
        output[:,:,y:y+args.patchsize,x:x+args.patchsize] += prediction
        layer1[:,:,y:y+args.patchsize,x:x+args.patchsize] += 1
          
    output /= layer1
    output  = output.squeeze()
    Image.fromarray(output).save("/export_afenfs_large-capacity/shared/iis/.DEBUG/will/building_classification/"+path+"/"+str(file_name))
    f.write(wdr)
    f.write(file_name)
    f.write('\n')
    
f.close()