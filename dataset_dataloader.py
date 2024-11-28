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
                 data_type: str, # slices / 3d_block / 3d_block_64 / 3d_block_128
                 do_augmentation: bool = False,
                 slices: bool = False):
        """Initialization."""
        self.root_imgs_dir = imgs_dir
        self.root_masks_dir = masks_dir
        self.df = df
        self.augmentations = get_augmentations(phase) #not used
        self.do_augmentation = False #do_augmentation
        self.slices = slices
        self.data_type = data_type
        print("DATATYPE: ", data_type)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        ct_id = self.df.loc[idx, "ID"]
        #mask_name = self.df.loc[idx, "MaskId"]


        if self.data_type == "slices" or self.data_type == "3d_block_V2":
            slice_group_index = self.df.loc[idx, "SliceIndex"]
        else:
            slice_group_index = self.df.loc[idx, "BlockIndex"]

        
        """
        if (self.slices):
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = np.expand_dims(mask, axis=2)"""



        """
        img = nib.load(os.path.join(self.root_imgs_dir, ct_id + '.nii.gz'))
        mask = nib.load(os.path.join(self.root_masks_dir, ct_id + '.nii.gz'))
        img = img.get_fdata()
        mask = mask.get_fdata()
        assert(img.shape == mask.shape)"""

        

        

        if self.data_type == "slices":
            img = np.load(os.path.join(self.root_imgs_dir, ct_id + '_' + str(slice_group_index) + '.npy'))
            mask = np.load(os.path.join(self.root_masks_dir, ct_id + '_' + str(slice_group_index) + '.npy' ))
            assert(img.shape == mask.shape)
            #padXY
        elif self.data_type == "3d_block":
            """
            first_slice = (slice_group_index * 32)
            end_index = min(first_slice + 32, img.shape[2])
            img = img[:, :, first_slice:end_index]
            mask = mask[:, :, first_slice:end_index]"""
            img = np.load(os.path.join(self.root_imgs_dir, ct_id + '_' + str(slice_group_index) + '.npy'))
            mask = np.load(os.path.join(self.root_masks_dir, ct_id + '_' + str(slice_group_index) + '.npy' ))
            assert(img.shape == mask.shape)
            if (img.shape[2] < 32):
                #padbottom
                required_padding = 32 - img.shape[2]
                img = np.pad(img, ((0, 0), (0, 0), (0, required_padding)), mode='constant', constant_values=-500) #MIN VALUE
                mask = np.pad(mask, ((0, 0), (0, 0), (0, required_padding)), mode='constant', constant_values=0.0) #MIN VALUE
            #pad X and Y
        elif self.data_type == "3d_block_64":
            img = np.load(os.path.join(self.root_imgs_dir, ct_id + '_' + str(slice_group_index) + '.npy'))
            mask = np.load(os.path.join(self.root_masks_dir, ct_id + '_' + str(slice_group_index) + '.npy' ))
            assert(img.shape == mask.shape)
            if (img.shape[2] < 64):
                #padbottom
                required_padding = 64 - img.shape[2]
                img = np.pad(img, ((0, 0), (0, 0), (0, required_padding)), mode='constant', constant_values=-500) #MIN VALUE
                mask = np.pad(mask, ((0, 0), (0, 0), (0, required_padding)), mode='constant', constant_values=0.0) #MIN VALUE
            #pad X and Y
        elif self.data_type == "3d_block_128":
            img = np.load(os.path.join(self.root_imgs_dir, ct_id + '_' + str(slice_group_index) + '.npy'))
            mask = np.load(os.path.join(self.root_masks_dir, ct_id + '_' + str(slice_group_index) + '.npy' ))
            assert(img.shape == mask.shape)
            if (img.shape[2] < 128):
                #padbottom
                required_padding = 128 - img.shape[2]
                img = np.pad(img, ((0, 0), (0, 0), (0, required_padding)), mode='constant', constant_values=-500) #MIN VALUE
                mask = np.pad(mask, ((0, 0), (0, 0), (0, required_padding)), mode='constant', constant_values=0.0) #MIN VALUE
        elif self.data_type == "3d_block_128FH":
            img = np.load(os.path.join(self.root_imgs_dir, ct_id + '.npy'))
            mask = np.load(os.path.join(self.root_imgs_dir, ct_id + '.npy'))

        else:
            raise Error("Unknown dataset type")


        if (max(img.shape[0], img.shape[1]) > 256):
            raise Error("CT slices can't fit into 256x256!")

        np.clip(img, -500, 500, img)
        #np.clip(mask, -500, 500, mask)


        #0-1
        img = (img + 500) / 1000
        

        target_xy = (256, 256)
        
        img = self.pad_XY(img, target_xy, 0)
        mask = self.pad_XY(mask, target_xy, 0)

        assert(img.shape == mask.shape)
        
        if self.data_type=="3d_block":
            assert(img.shape == (256, 256, 32))
        elif self.data_type=="3d_block_64":
            assert(img.shape == (256, 256, 64))
        elif self.data_type == "3d_block_128" or self.data_type == "3d_block_128FH":
            assert(img.shape == (256, 256, 128))
        else:
            assert(img.shape ==(256, 256))

        


        img = np.expand_dims(img, axis=0)
        mask = np.expand_dims(mask, axis=0)

        img = img.astype(np.float32)
        mask = mask.astype(np.float32)


        #nem használok augmentation -t
        """
        if self.do_augmentation:
            augmented = self.augmentations(image=img,
                                        mask=mask.astype(np.float32))
            img = augmented['image']
            mask = augmented['mask'].permute(2, 0, 1)"""


        return img, mask, ct_id, slice_group_index

    def pad_XY(self, arr, target_xy, value):

        # Original dimensions
        x = arr.shape[0]
        y = arr.shape[1]


        # Calculate padding sizes
        pad_x_before = (target_xy[0] - x) // 2
        pad_x_after = target_xy[0] - x - pad_x_before

        pad_y_before = (target_xy[1] - y) // 2
        pad_y_after = target_xy[1] - y - pad_y_before

        paddings = ((pad_x_before, pad_x_after), (pad_y_before, pad_y_after), (0, 0));
        paddings = paddings[:arr.ndim] 

        # Apply padding
        padded_array = np.pad(
            arr,
            pad_width=paddings,  # Pad only X and Y
            mode='constant',  # Use constant padding (default value is 0)
            constant_values=value
        )

        return padded_array

    def pad_array(self, arr, target_shape):
        # Padding value
        padding_value = 0

        # Compute padding for each dimension
        """pad_width = [(0, 0)] * arr.ndim
        for i, (original_dim, desired_dim) in enumerate(zip(arr.shape, target_shape)):
            total_pad = max(desired_dim - original_dim, 0)
            pad_before = total_pad // 2
            pad_after = total_pad - pad_before
            pad_width[i] = (pad_before, pad_after)"""


        pad_width = [
            (max((target_shape[0] - arr.shape[0]) // 2, 0),
            max((target_shape[0] - arr.shape[0] + 1) // 2, 0)),
            (max((target_shape[1] - arr.shape[1]) // 2, 0),
            max((target_shape[1] - arr.shape[1] + 1) // 2, 0)),
            (0, 0)  # No padding along the third axis
            #(max((target_shape[2] - arr.shape[2]) // 2, 0),
            #max((target_shape[2] - arr.shape[2] + 1) // 2, 0))
        ]

        # Create the new array with padding
        new_array = np.pad(arr, pad_width, constant_values=padding_value)

        assert(new_array.shape[0] == target_shape[0] and new_array.shape[1] == target_shape[1])

        return new_array 

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
            #Normalize(mean=mean, std=std, p=1),
            #ToTensor(num_classes=3, sigmoid=False),
            ToTensorV2(),
        ]
    )
    list_trfms = Compose(list_transforms)
    return list_trfms


def get_dataloader(
    imgs_dir: str,
    masks_dir: str,
    train_csv: str,
    val_csv: str,
    phase: str,
    batch_size: int = 8,
    num_workers: int = 2,
    test_csv: str = "",    
    #test_size: float = 0.2,
    data_type: str = "slices", # # slices / 3d_block / 3d_block_V2,
    shuffle: bool = True
):
    '''Returns: dataloader for the model training'''
    """df = pd.read_csv(train_csv)


   

    train_df, val_df = train_test_split(df,
                                          test_size=test_size,
                                          random_state=69)
    train_df, val_df = train_df.reset_index(drop=True), val_df.reset_index(drop=True)"""


    if phase == "train":
        df = pd.read_csv(train_csv, sep=";")
        df.reset_index(drop=True)
    elif phase == "test":
        df = pd.read_csv(test_csv, sep=";")
        df.reset_index(drop=True)
    else:
        df = pd.read_csv(val_csv, sep=";")
        df.reset_index(drop=True)

    # df = train_df if phase == "train" else val_df

    #determinisztikusan random
    #df.to_csv("val1.csv")



    image_dataset = LungsDataset(imgs_dir, masks_dir, df, phase, data_type)
    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=shuffle,
    )

    return dataloader
