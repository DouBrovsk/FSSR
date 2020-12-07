import imghdr
import sys
from PIL import Image
import os
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

#import Augmentor  # Data augmentation library

os.chdir(os.path.dirname(os.path.realpath(__file__)))


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

        
        print(query_label.size())
        print(query_data.size())
        
        
        del original
        if self.mode == 'train':
            print(torch.stack(support_data).size())
            print(torch.stack(support_label).size())
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
    
