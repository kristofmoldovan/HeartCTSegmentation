import os

import numpy as np
import pandas as pd
import cv2

import scipy.ndimage

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from albumentations.pytorch import ToTensor, ToTensorV2
from albumentations import (HorizontalFlip,
                            VerticalFlip,
                            Normalize,
                            Compose)

import nibabel as nib


class LungsDataset(Dataset):
    def __init__(self,
                 imgs_dir: str,
                 masks_dir:str,
                 df: pd.DataFrame,
                 phase: str,
                 do_augmentation: bool = False,
                 slices: bool = False):
        """Initialization."""
        self.root_imgs_dir = imgs_dir
        self.root_masks_dir = masks_dir
        self.df = df
        self.augmentations = get_augmentations(phase)
        self.do_augmentation = do_augmentation
        self.slices = slices

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, "ImageId"]
        mask_name = self.df.loc[idx, "MaskId"]
        img_path = os.path.join(self.root_imgs_dir, img_name)
        mask_path = os.path.join(self.root_masks_dir, mask_name)
        
        
        if (self.slices):
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = np.expand_dims(mask, axis=2)

        else:
            img = nib.load(os.path.join(self.root_imgs_dir, img_name + '.nii.gz'))
            mask = nib.load(os.path.join(self.root_masks_dir, mask_name + '.nii.gz'))
            img = img.get_fdata()
            mask = mask.get_fdata()

        #print(mask_path)
        #print("mask type: ", str(type(mask)))

        #with open('file.txt', 'a') as f:
        #    f.write("PATH: " + mask_path + " TYPE " + str(type(mask)) + "\n")

        mask[mask < 240] = 0    # remove artifacts
        mask[mask > 0] = 1

        target_shape = (256, 256, 256)

        

        img = self.expand_3d_array(img, target_shape)
        mask = self.expand_3d_array(mask, target_shape, True)

        img = np.expand_dims(img, axis=0)
        mask = np.expand_dims(mask, axis=0)

        img = img.astype(np.float32)
        mask = mask.astype(np.float32)

        print("IMG SHAPE: ", img.shape)
        print("mask shape: ", mask.shape)

        print(img.shape)
        print(mask.shape)

        if self.do_augmentation:
            augmented = self.augmentations(image=img,
                                        mask=mask.astype(np.float32))
            img = augmented['image']
            mask = augmented['mask'].permute(2, 0, 1)

        return img, mask#, img_name

    def expand_3d_array(self, arr, target_shape, is_mask: bool = False):
        x, y, z = arr.shape
        tx, ty, tz = target_shape

        print(x, y, z)

        # Scale factors
        scale_factor = min(tx/x, ty/y, tz/z)

        print(scale_factor)

        if (is_mask):
            order = 0
        else:
            order = 1

        # Linear interpolation (order=1)
        scaled_arr = scipy.ndimage.zoom(arr, scale_factor, order=order)

        print(scaled_arr.shape)

        sx, sy, sz = scaled_arr.shape

        pad_x = int((tx - sx) // 2)
        pad_y = int((ty - sy) // 2)
        pad_z = int((tz - sz) // 2)

        pad_x2 = tx - sx - pad_x
        pad_y2 = ty - sy - pad_y
        pad_z2 = tz - sz - pad_z


        padded_arr = np.pad(scaled_arr, ((pad_x, pad_x2), (pad_y, pad_y2), (pad_z, pad_z2)), mode='constant', constant_values=0)

        print(padded_arr.shape)

        return padded_arr
    


def get_augmentations(phase,
                   mean: tuple = (0.485, 0.456, 0.406),
                   std: tuple = (0.229, 0.224, 0.225),):
    list_transforms = []
    if phase == "train":
        list_transforms.extend(
            [
                VerticalFlip(p=0.5),
            ]
        )
    list_transforms.extend(
        [
            Normalize(mean=mean, std=std, p=1),
            #ToTensor(num_classes=3, sigmoid=False),
            ToTensorV2(),
        ]
    )
    list_trfms = Compose(list_transforms)
    return list_trfms


def get_dataloader(
    imgs_dir: str,
    masks_dir: str,
    path_to_csv: str,
    phase: str,
    batch_size: int = 8,
    num_workers: int = 2,
    test_size: float = 0.2,
):
    '''Returns: dataloader for the model training'''
    df = pd.read_csv(path_to_csv)


   

    train_df, val_df = train_test_split(df,
                                          test_size=test_size,
                                          random_state=69)
    train_df, val_df = train_df.reset_index(drop=True), val_df.reset_index(drop=True)

    df = train_df if phase == "train" else val_df

    #determinisztikusan random
    #df.to_csv("val1.csv")

    image_dataset = LungsDataset(imgs_dir, masks_dir, df, phase)
    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )

    return dataloader
