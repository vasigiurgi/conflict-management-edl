#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 14:28:32 2021

@author: deeplearning-miam
"""
import os
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from skimage.transform import rotate

def img_path (dir, ext):
    # Path to files
    img_path=sorted(
    [
        os.path.join(dir, fname)
        for fname in os.listdir(dir)
        if fname.endswith(ext) and fname.startswith("u")
    ]
)
    return img_path

# category label
def cat_label (paths_list):
    path_label=[] # 0 for urban marked (um), 1 for urban multiple marked (umm) and 2 for urban unmarked (uu)
    for i in range(len(paths_list)):
        fname= os.path.split(paths_list[i])[1]
        if fname.startswith ("um_"):
            path_label.append(0)
        elif fname.startswith("umm_"):
            path_label.append(1)
        elif fname.startswith("uu_"):
            path_label.append(2)
    return path_label
        
# def train_val_split(input_dir_rgb, input_dir_velo, target_dir):
#     """
#     Parameters
#     ----------
    
#     input_dir_rgb : string
#         DESCRIPTION. path to the rgb camera image
#     input_dir_velo : string
#         DESCRIPTION. path to the projected velodyne 
#     target_dir : string
#         DESCRIPTION. path to the ground truth image

#     Returns
#     -------
#     split_dic : dictionary
#         DESCRIPTION. dictionary of 10 train/val split for rgb, velo, target
#     """
#     split_dic={}
#     # Prepare the image paths
#     #RGB
#     img_paths_rgb= img_path(input_dir_rgb, ".png")
#     img_paths_rgb= np.array(img_paths_rgb)
#     # Velo
#     img_paths_velo= img_path(input_dir_velo, ".png")  
#     img_paths_velo= np.array(img_paths_velo)
#     # Target (ground truth)
#     target_img_paths= img_path(target_dir, ".png")
#     target_img_paths= np.array(target_img_paths)
#     # Path label
#     label= cat_label(img_paths_rgb)
    
#     skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2)
#     # camera split
#     i=1
#     for train_index, val_index in skf.split(img_paths_rgb, label):
#         train_name= "train_cam_" + str(i)
#         val_name= "val_cam_" + str(i)
#         split_dic[train_name]=list(shuffle(img_paths_rgb[train_index], random_state=0))
#         split_dic[val_name]= list(shuffle(img_paths_rgb[val_index], random_state=0))
#         i+=1
#     # velo split
#     i=1
#     for train_index, val_index in skf.split(img_paths_velo, label):
#         train_name= "train_velo_" + str(i)
#         val_name= "val_velo_" + str(i)
#         split_dic[train_name]= list(shuffle(img_paths_velo[train_index], random_state=0))
#         split_dic[val_name]= list(shuffle(img_paths_velo[val_index], random_state=0))
#         i+=1
#     # target(groud truth) split
#     i=1
#     for train_index, val_index in skf.split(target_img_paths, label):
#         train_name= "train_target_" + str(i)
#         val_name= "val_target_" + str(i)
#         split_dic[train_name]= list(shuffle(target_img_paths[train_index], random_state=0))
#         split_dic[val_name]= list(shuffle(target_img_paths[val_index], random_state=0))
#         i+=1
        
#     return split_dic
# stratified 10-fold cross-validation
def train_val_split(input_dir_rgb, input_dir_velo, target_dir, random_state=1):
    """
    Parameters
    ----------
    input_dir_rgb : string
        DESCRIPTION. path to the rgb camera image
    input_dir_velo : string
        DESCRIPTION. path to the projected velodyne 
    target_dir : string
        DESCRIPTION. path to the ground truth image
    random_state : int, optional
        DESCRIPTION. random state for reproducibility (default is 1)

    Returns
    -------
    split_dic : dictionary
        DESCRIPTION. dictionary of 10 train/val splits for rgb, velo, target
    """
    split_dic = {}
    # Prepare the image paths
    # RGB
    img_paths_rgb = img_path(input_dir_rgb, ".png")
    img_paths_rgb = np.array(img_paths_rgb)
    # Velo
    img_paths_velo = img_path(input_dir_velo, ".png")  
    img_paths_velo = np.array(img_paths_velo)
    # Target (ground truth)
    target_img_paths = img_path(target_dir, ".png")
    target_img_paths = np.array(target_img_paths)
    # Path label
    label = cat_label(img_paths_rgb)
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    
    # Camera split
    i = 1
    for train_index, val_index in skf.split(img_paths_rgb, label):
        train_name = "train_cam_" + str(i)
        val_name = "val_cam_" + str(i)
        split_dic[train_name] = list(shuffle(img_paths_rgb[train_index], random_state=0))
        split_dic[val_name] = list(shuffle(img_paths_rgb[val_index], random_state=0))
        i += 1
    
    # Velo split
    i = 1
    for train_index, val_index in skf.split(img_paths_velo, label):
        train_name = "train_velo_" + str(i)
        val_name = "val_velo_" + str(i)
        split_dic[train_name] = list(shuffle(img_paths_velo[train_index], random_state=0))
        split_dic[val_name] = list(shuffle(img_paths_velo[val_index], random_state=0))
        i += 1
    
    # Target (ground truth) split
    i = 1
    for train_index, val_index in skf.split(target_img_paths, label):
        train_name = "train_target_" + str(i)
        val_name = "val_target_" + str(i)
        split_dic[train_name] = list(shuffle(target_img_paths[train_index], random_state=0))
        split_dic[val_name] = list(shuffle(target_img_paths[val_index], random_state=0))
        i += 1
        
    return split_dic


def split_check(paths_list):
    #split a path in to file keys for preparing split checking
    
    file_names=[]
    for item in  paths_list:
        file_key=os.path.split(item)[1].split('.')[0]
        f_split=file_key.split('_')
        if 'road' in f_split:
            file_key= f_split[0]+ '_' + f_split[2]
        else:
            pass
        file_names.append(file_key)
    return file_names



def write_path(out_dir,paths_list,file_name):
    # Write the validation paths to text file
    path_name=os.path.join(out_dir,file_name)
    with open(path_name, 'w') as f:
      # write elements of list
      for items in paths_list:
          f.write('%s\n' %items)
    # close the file
    f.close()
    return 0

#Prepare Sequence class to load & vectorize batches of data: individual random aug, not at epch end
class kittiroad(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths_rgb,input_img_paths_velo, target_img_paths,val=False):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths_rgb = input_img_paths_rgb
        self.input_img_paths_velo = input_img_paths_velo
        self.target_img_paths = target_img_paths
        self.val=val

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths_rgb = self.input_img_paths_rgb[i : i + self.batch_size]
        batch_input_img_paths_velo=self.input_img_paths_velo[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x1 = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        angle=random.uniform(-20, 20)
        for j, path in enumerate(batch_input_img_paths_rgb):
            img = load_img(path)
            w,h=img.size
            dh=int((self.img_size[0]-h)/2)
            dw=int((self.img_size[1]-w)/2)
            zero_padded_img=np.zeros(self.img_size+(3,), dtype="float32")
            zero_padded_img[dh:dh+h,dw:dw+w,:]=img
            zero_padded_img/=255.0
            if self.val==False: # training
                zero_padded_img=rotate(zero_padded_img,angle)
            x1[j] = zero_padded_img
            
        x2 = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")   
        for j, path in enumerate(batch_input_img_paths_velo):
            img = load_img(path)
            w,h=img.size
            dh=int((self.img_size[0]-h)/2)
            dw=int((self.img_size[1]-w)/2)
            zero_padded_img=np.zeros(self.img_size+(3,), dtype="float32")
            zero_padded_img[dh:dh+h,dw:dw+w,:]=img
            zero_padded_img/=255.0
            if self.val==False: # training 
                zero_padded_img=rotate(zero_padded_img,angle)
            x2[j] = zero_padded_img
            
        x1_x2=[x1,x2]   
        
        y = np.zeros((self.batch_size,) + self.img_size + (2,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path)
            w,h=img.size
            dh=int((self.img_size[0]-h)/2)
            dw=int((self.img_size[1]-w)/2)
            zero_padded_img=np.zeros(self.img_size+(3,), dtype="uint8")
            zero_padded_img[dh:dh+h,dw:dw+w,:]=img
            zero_padded_img= zero_padded_img/255
            if self.val==False:  # training
                zero_padded_img=rotate(zero_padded_img,angle)
            img=zero_padded_img>0
            
            # Two labels: ch-0: not-road, ch-1: road [i.e. not-road: m(theta_1)=1,m(theta_2)=0; road: m(theta_1)=0,m(theta_2)=1]
            label_img= img[:,:,1:]
            label_img[:,:,0]=~label_img[:,:,1] 
            
            y[j]=label_img
            
        return x1_x2, y
            
            
    
