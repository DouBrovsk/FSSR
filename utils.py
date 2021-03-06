import imghdr
import sys
from PIL import Image
import os
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2lab, lab2rgb
import sklearn.neighbors as nn

#import Augmentor  # Data augmentation library

os.chdir(os.path.dirname(os.path.realpath(__file__)))


def encode_313bin(data_ab_ss, nn_enc):
    '''Encode to 313bin
    Args:
    data_ab_ss: [N, H, W, 2]
    Returns:
    gt_ab_313 : [N, H, W, 313]
    '''

    data_ab_ss = np.transpose(data_ab_ss, (0, 3, 1, 2))
    gt_ab_313 = nn_enc.encode_points_mtx_nd(data_ab_ss, axis=1)

    gt_ab_313 = np.transpose(gt_ab_313, (0, 2, 3, 1))
    return gt_ab_313
## Utils
def show_image(input_tensor, n=0):
    y = input_tensor.detach()[n].cpu().numpy().transpose((1, 2, 0))
    plt.imshow(y)
    plt.pause(1)


def is_file_not_corrupted(path):
    return (imghdr.what(path) == 'jpeg' or imghdr.what(
        path) == 'png')  # Checks the first bytes of the file to see if it's a valid png/jpeg
    
class Logger(object):
    def __init__(self,name):
        self.file=open(name,'a')
        
    def write(self,msg):
        sys.stdout.write(msg)
        self.file.write(msg)
        
    def flush(self):
        pass
        
    def __del__(self):
        self.file.close()


class DADataset(torch.utils.data.Dataset):  # Making artificial tasks with Data Augmentation. It's very bad if used for validation because it means the validation set is changing at every epoch -> Refrain from using this for validation.
    def __init__(self, images_directory, transform, num_shot, is_valid_file=is_file_not_corrupted, scale_factor=2,
                 memory_fit_factor=4, mode='train'):
        self.is_valid_file = is_valid_file
        self.image_paths = [os.path.join(images_directory, f) for f in os.listdir(images_directory) if
                            self.is_valid_file(os.path.join(images_directory, f))]
        self.length = len(self.image_paths)
        self.transform = transform
        self.scale_factor = scale_factor
        self.num_shot = num_shot
        self.memfact = memory_fit_factor
        self.mode = mode

    def __getitem__(self, index):
        
        
        original = Image.open(self.image_paths[index]).convert('RGB')
        width, height = original.width, original.height
        #if self.mode == 'train':
        #    resize_height = height // self.memfact
        #    resize_width = width // self.memfact
        #    while resize_height * resize_width > 393 * 510:  # Spaghetis to avoid too big tensors so it fits into 1 GPU.
        #        resize_height -= self.memfact
        #        resize_width -= int(self.memfact * (width / height))
        #else:
        #resize_height = height//4
        #resize_width = width//4
        resize_height = 200
        resize_width = 200
        

        if resize_height % self.scale_factor != 0:
            resize_height -= (resize_height % self.scale_factor)
        if resize_width % self.scale_factor != 0:
            resize_width -= (resize_width % self.scale_factor)

        # ToDo: DATA AUGMENTATION WITH THE LIBRARY

        query_label = self.transform(
            transforms.Resize((resize_height, resize_width), interpolation=Image.BICUBIC)(original))
        
        query_data = self.transform(
            transforms.Resize((resize_height // self.scale_factor, resize_width // self.scale_factor),
                              interpolation=Image.BICUBIC)(original))  # ToDo: Change code to make the set more customizable?
        
        support_label, support_data = [], []
        
        augmentation = transforms.Compose(
            [transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=(-0.1, 0.1)),
             transforms.RandomPerspective(), 
             transforms.RandomRotation((-15, 15)),
             #transforms.RandomResizedCrop((resize_height, resize_width),scale=(0.5, 0.9), interpolation=Image.BICUBIC),
             transforms.RandomGrayscale(p=0.02), 
             transforms.RandomHorizontalFlip(0.3),
             transforms.RandomVerticalFlip(0.3)])
            
        for i in range(self.num_shot):
            transformed_img = augmentation(original) # chaque appel de augmentation redefinit une nouvelle transformation !
            
            support_label.append(self.transform(
                transforms.Resize((resize_height, resize_width),
                                  interpolation=Image.BICUBIC)(transformed_img)))
            
            support_data.append(self.transform(
                transforms.Resize((resize_height // self.scale_factor, resize_width // self.scale_factor),
                                  interpolation=Image.BICUBIC)(transformed_img)))

        #print('task')
        #print(query_label.size())
        #print(query_data.size())
        #print(torch.stack(support_data).size())
        #print(torch.stack(support_label).size())
        
        
        del original
        if self.mode == 'train':
            return torch.stack(support_data), torch.stack(support_label), query_data, query_label
        elif self.mode == 'up':
            return torch.stack(support_data), torch.stack(support_label), query_label
        else:
            raise NotImplementedError

    def __len__(self):
        return self.length


class FSDataset(torch.utils.data.Dataset):
    '''
    Assuming classes_folder_path is a directory containing folders of N images of the same category,
    We take N-1 images for the support set and 1 image for the query set.
    The support set is composed of N-1 couples of images (downsampled_image, original_image).
    The query set is composed of 1 image couple (downsampled_image, original image).
    In this setup, it is a N-1 shot (1 way) super-resolution task.
    '''

    def __init__(self, classes_folder_path, transform, is_valid_file=is_file_not_corrupted, scale_factor=2, mode='train'):
        self.is_valid_file = is_valid_file
        self.class_paths = [os.path.join(classes_folder_path, f) for f in os.listdir(classes_folder_path) if os.path.isdir(os.path.join(classes_folder_path, f)) and len(os.listdir(os.path.join(classes_folder_path, f))) > 2]
        self.length = len(self.class_paths)
        self.transform = transform
        self.mode = mode
        self.scale_factor = scale_factor

    def __getitem__(self, index):  # ToDO: Implement the method.
        transform = transforms.ToTensor()
        folder = self.class_paths[index]
        files = [os.path.join(folder, f) for f in os.listdir(folder)]
        support, support_l = [], []
        for i in range(len(files) - 1):
            img = Image.open(files[i])
            resize_width, resize_height = img.width, img.height
            if resize_height % self.scale_factor != 0:
                resize_height -= (resize_height % self.scale_factor)
            if resize_width % self.scale_factor != 0:
                resize_width -= (resize_width % self.scale_factor)
            support_l.append(transform(img))
            support.append(transform(transforms.Resize((resize_height//self.scale_factor, resize_width//self.scale_factor), interpolation=Image.BICUBIC)(img)))
        support = torch.stack(support)
        support_l = torch.stack(support_l)

        img = Image.open(files[-1])
        resize_width, resize_height = img.width, img.height
        if resize_height % self.scale_factor != 0:
            resize_height -= (resize_height % self.scale_factor)
        if resize_width % self.scale_factor != 0:
            resize_width -= (resize_width % self.scale_factor)
        query_l = transform(img)
        query = transform(transforms.Resize((resize_height//self.scale_factor, resize_width//self.scale_factor), interpolation=Image.BICUBIC)(img))
        if self.mode == 'train':
            return support, support_l, query, query_l
        elif self.mode == 'up':
            return support, support_l, query
        else:
            raise NotImplementedError

    def __len__(self):
        return self.length
    
    
class NNEncode():
    ''' Encode points using NN search and Gaussian kernel '''
    def __init__(self,NN,sigma,km_filepath='',cc=-1):
        
        self.cc = np.load(km_filepath)
        '''
        if(check_value(cc,-1)):
            self.cc = np.load(km_filepath)
        else:
            self.cc = cc
        '''
        self.K = self.cc.shape[0]
        self.NN = int(NN)
        self.sigma = sigma
        self.nbrs = nn.NearestNeighbors(n_neighbors=self.NN, algorithm='ball_tree').fit(self.cc)

        self.alreadyUsed = False

    def encode_points_mtx_nd(self,pts_nd,axis=1,returnSparse=False,sameBlock=True):

        pts_flt = flatten_nd_array(pts_nd,axis=axis)

        P = pts_flt.shape[0]
        if(sameBlock and self.alreadyUsed):
            self.pts_enc_flt[...] = 0 # already pre-allocated
        else:
            self.alreadyUsed = True
            self.pts_enc_flt = np.zeros((P,self.K))
            self.p_inds = np.arange(0,P,dtype='int')[:,na()]

        P = pts_flt.shape[0]

        (dists,inds) = self.nbrs.kneighbors(pts_flt)

        wts = np.exp(-dists**2/(2*self.sigma**2))
        wts = wts/np.sum(wts,axis=1)[:,na()]

        self.pts_enc_flt[self.p_inds,inds] = wts
        pts_enc_nd = unflatten_2d_array(self.pts_enc_flt,pts_nd,axis=axis)

        return pts_enc_nd

    def decode_points_mtx_nd(self,pts_enc_nd,axis=1):
        pts_enc_flt = flatten_nd_array(pts_enc_nd,axis=axis)
        pts_dec_flt = np.dot(pts_enc_flt,self.cc)
        pts_dec_nd = unflatten_2d_array(pts_dec_flt,pts_enc_nd,axis=axis)
        return pts_dec_nd

    def decode_1hot_mtx_nd(self,pts_enc_nd,axis=1,returnEncode=False):
        pts_1hot_nd = nd_argmax_1hot(pts_enc_nd,axis=axis)
        pts_dec_nd = self.decode_points_mtx_nd(pts_1hot_nd,axis=axis)
        if(returnEncode):
            return (pts_dec_nd,pts_1hot_nd)
        else:
            return pts_dec_nd
    
def unflatten_2d_array(pts_flt,pts_nd,axis=1,squeeze=False):
    ''' Unflatten a 2d array with a certain axis
    INPUTS
        pts_flt     prod(N \ N_axis) x M array
        pts_nd      N0xN1x...xNd array
        axis        integer
        squeeze     bool     if true, M=1, squeeze it out
    OUTPUTS
        pts_out     N0xN1x...xNd array        '''
    NDIM = pts_nd.ndim
    SHP = np.array(pts_nd.shape)
    nax = np.setdiff1d(np.arange(0,NDIM),np.array((axis))) # non axis indices
    NPTS = np.prod(SHP[nax])

    if(squeeze):
        axorder = nax
        axorder_rev = np.argsort(axorder)
        M = pts_flt.shape[1]
        NEW_SHP = SHP[nax].tolist()
        # print NEW_SHP
        # print pts_flt.shape
        pts_out = pts_flt.reshape(NEW_SHP)
        pts_out = pts_out.transpose(axorder_rev)
    else:
        axorder = np.concatenate((nax,np.array(axis).flatten()),axis=0)
        axorder_rev = np.argsort(axorder)
        M = pts_flt.shape[1]
        NEW_SHP = SHP[nax].tolist()
        NEW_SHP.append(M)
        pts_out = pts_flt.reshape(NEW_SHP)
        pts_out = pts_out.transpose(axorder_rev)

    return pts_out

def flatten_nd_array(pts_nd,axis=1):
    ''' Flatten an nd array into a 2d array with a certain axis
    INPUTS
        pts_nd       N0xN1x...xNd array
        axis         integer
    OUTPUTS
        pts_flt     prod(N \ N_axis) x N_axis array     '''
    NDIM = pts_nd.ndim
    SHP = np.array(pts_nd.shape)
    nax = np.setdiff1d(np.arange(0,NDIM),np.array((axis))) # non axis indices
    NPTS = np.prod(SHP[nax])
    axorder = np.concatenate((nax,np.array(axis).flatten()),axis=0)
    pts_flt = pts_nd.transpose((axorder))
    pts_flt = pts_flt.reshape(NPTS,SHP[axis])
    return pts_flt

def na(): # shorthand for new axis
    return np.newaxis
    
    
class ColorDataset(torch.utils.data.Dataset):
    '''
    Assuming classes_folder_path is a directory containing folders of N images of the same category,
    We take N-1 images for the support set and 1 image for the query set.
    The support set is composed of N-1 couples of images (greyscale_image, original_image).
    The query set is composed of 1 image couple (greyscale_image, original image).
    In this setup, it is a N-1 shot (1 way) colorization task.
    '''

    def __init__(self, classes_folder_path):
        
        self.class_paths = [os.path.join(classes_folder_path, f) for f in os.listdir(classes_folder_path) if os.path.isdir(os.path.join(classes_folder_path, f)) and len(os.listdir(os.path.join(classes_folder_path, f))) > 2]
        self.length = len(self.class_paths)
        self.nnenc = NNEncode(10, 5, km_filepath = "./pts_in_hull.npy")
        

    def __getitem__(self, index):  # ToDO: Implement the method.
        
        transform = transforms.ToTensor()
        folder = self.class_paths[index]
        files = [os.path.join(folder, f) for f in os.listdir(folder)]
        support, support_l = [], []
        
        for i in range(len(files) - 1):
            
            rgb_image = Image.open(files[i]).convert('RGB')
            w, h = rgb_image.size
            if w != h:
                min_val = min(w, h)
                rgb_image = rgb_image.crop((w // 2 - min_val // 2, h // 2 - min_val // 2, w // 2 + min_val // 2, h // 2 + min_val // 2))
        
            rgb_image = np.array(rgb_image.resize((256,256), Image.LANCZOS))
        
            lab_image = rgb2lab(rgb_image)
            l_image = lab_image[:,:,:1]
            ab_image = lab_image[:,:,1:]
        
            
            color_feat = encode_313bin(np.expand_dims(ab_image, axis = 0), self.nnenc)[0]
            color_feat = np.mean(color_feat, axis = (0, 1))
            
            support_l.append(transform(rgb_image))
            
            gray_image = [lab_image[:,:,:1]]
            h, w, c = lab_image.shape
            gray_image.append(np.zeros(shape = (h, w, 2)))
            gray_image = np.concatenate(gray_image, axis = 2)
        
            support.append(transform(gray_image))
            
        support = torch.stack(support)
        support_l = torch.stack(support_l)


        rgb_image = Image.open(files[-1]).convert('RGB')
        w, h = rgb_image.size
        if w != h:
            min_val = min(w, h)
            rgb_image = rgb_image.crop((w // 2 - min_val // 2, h // 2 - min_val // 2, w // 2 + min_val // 2, h // 2 + min_val // 2))
    
        rgb_image = np.array(rgb_image.resize((256,256), Image.LANCZOS))
    
        lab_image = rgb2lab(rgb_image)
        l_image = lab_image[:,:,:1]
        ab_image = lab_image[:,:,1:]
    
        i
        color_feat = encode_313bin(np.expand_dims(ab_image, axis = 0), self.nnenc)[0]
        color_feat = np.mean(color_feat, axis = (0, 1))
        
        query_l = (transform(rgb_image))
        
        gray_image = [lab_image[:,:,:1]]
        h, w, c = lab_image.shape
        gray_image.append(np.zeros(shape = (h, w, 2)))
        gray_image = np.concatenate(gray_image, axis = 2)
    
        query = (transform(gray_image))
        
        
        return support, support_l, query, query_l
        


    def __len__(self):
        return self.length
    
        
    
