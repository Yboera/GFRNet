import  os
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from PIL import Image, ImageEnhance
import random
import numpy as np
import scipy.io as sio
import cv2
import torch
from skimage import segmentation, color
import matplotlib.pyplot as plt

def show(allfocus, fs, depth, gt):
        plt.imshow(allfocus[0])
        plt.show()
        plt.imshow(depth[0])
        plt.show()
        plt.imshow(fs[0][0])
        plt.show()
        plt.imshow(fs[11][0])
        plt.show()
        plt.imshow(gt[0])
        plt.show()
        
def randomCrop(image, fss, depth, label):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width-border , image_width)
    crop_win_height = np.random.randint(image_height-border , image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    num_fs = len(fss)
    for idx in range(num_fs):
        fss[idx] = fss[idx].crop(random_region)
    
    return image.crop(random_region), fss, depth.crop(random_region), label.crop(random_region) 

def randomRotation(image, fss, depth, label):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        depth = depth.rotate(random_angle, mode)
        num_fs = len(fss)
        for idx in range(num_fs):
            fss[idx] = fss[idx].rotate(random_angle, mode)
    return image, fss, depth, label 

def colorEnhance(image):
    bright_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity=random.randint(0,20)/10.0
    image=ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity=random.randint(0,30)/10.0
    image=ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image

def randomPeper(img):
    #img=np.asarray(img)
    img=np.array(img)  #python 3.8要求
    noiseNum=int(0.0015*img.shape[0]*img.shape[1])
    for i in range(noiseNum):
        randX=random.randint(0,img.shape[0]-1)
        randY=random.randint(0,img.shape[1]-1)
        if random.randint(0,1)==0:
            img[randX,randY]=0
        else:
            img[randX,randY]=255 
    return Image.fromarray(img)  

class ALLDataset(Dataset):
    def __init__(self, location = None, train = True, dataEnhance = True):
        self.location = location
        self.num = len(os.listdir(self.location+ 'allfocus/'))
        self.train = train
        self.dataEnhance = dataEnhance

        self.allfocus_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()])
        self.depth_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()])
        self.fs_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        self.allfocus_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_test = transforms.Compose([
            transforms.ToTensor()])
        self.depth_test = transforms.Compose([
            transforms.ToTensor()])
        self.fs_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # self.allfocus_test = transforms.Compose([
        #     transforms.ToTensor()])
        # self.gt_test = transforms.Compose([
        #     transforms.ToTensor()])
        # self.depth_test = transforms.Compose([
        #     transforms.ToTensor()])
        # self.fs_test = transforms.Compose([
        #     transforms.ToTensor()])

    def __len__(self):
        return  len(os.listdir(self.location + 'allfocus/'))

    def __getitem__(self, idx):
        img_name = os.listdir(self.location + 'allfocus/')[idx]
        
        allfocus = Image.open(self.location + 'allfocus/' + img_name)
        allfocus = allfocus.convert('RGB')
        allfocus = allfocus.resize((256, 256))
       
        depth = Image.open(self.location + 'new_ndepth/' + img_name.split('.')[0] + '.png')
        depth = depth.convert('RGB')
        depth = depth.resize((256, 256))

        focalstack = sio.loadmat(self.location + 'mat/' + img_name.split('.')[0] + '.mat')
        focal_img = focalstack['img']
        #focal = np.asarray(focal,dtype=np.float32)/255.0
        focal_num = focal_img.shape[2] // 3
        focal = []
        for i in range(focal_num):
            img_PIL = Image.fromarray(np.uint8(focal_img[:, :, i*3:i*3+3]))
            img_PIL = img_PIL.resize((256, 256))
            focal.append(img_PIL)
        #new_fs = np.concatenate(temp_focal, axis=2)

        if self.train:
            GT = Image.open(self.location + 'GT/' + img_name.split('.')[0] + '.png')
            GT = GT.convert('L')
            GT = GT.resize((256,256))
        
            if self.dataEnhance:
                allfocus, focal, depth, GT = randomCrop(allfocus, focal, depth, GT)
                allfocus, focal, depth, GT = randomRotation(allfocus, focal, depth, GT)
                allfocus = colorEnhance(allfocus)
                GT = randomPeper(GT)

                allfocus = self.allfocus_transform(allfocus)
                GT = self.gt_transform(GT)
                depth = self.depth_transform(depth)
                for idx in range(focal_num):
                    focal[idx] = self.fs_transform(focal[idx])
                #show(allfocus, focal, depth, GT)    

            return allfocus, depth, focal, GT, img_name
        
        else:
            if self.dataEnhance:     
                allfocus = self.allfocus_test(allfocus)
                depth = self.depth_test(depth)
                for idx in range(focal_num):
                        focal[idx] = self.fs_test(focal[idx])

            return allfocus, depth, focal, img_name


