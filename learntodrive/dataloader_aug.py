import cv2
import numpy as np
import random

import os
from PIL import Image
import pandas as pd
import numpy as np
from random import shuffle
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
import torch

def fliph_image(img, mask):
    """
    Returns a horizontally flipped image
    """
    return cv2.flip(img, 1), cv2.flip(mask, 1)

def change_image_brightness_rgb(img, s_low=0.2, s_high=0.75):
    """
    Changes the image brightness by multiplying all RGB values by the same scalacar in [s_low, s_high).
    Returns the brightness adjusted image in RGB format.
    """
    img = img.astype(np.float32)
    s = np.random.uniform(s_low, s_high)
    img[:,:,:] *= s
    np.clip(img, 0, 255)
    return  img.astype(np.uint8)

def add_random_shadow(img, w_low=0.6, w_high=0.85):
    """
    Overlays supplied image with a random shadow polygon
    The weight range (i.e. darkness) of the shadow can be configured via the interval [w_low, w_high)
    """
    cols, rows = (img.shape[0], img.shape[1])
    
    top_y = np.random.random_sample() * rows
    bottom_y = np.random.random_sample() * rows
    bottom_y_right = bottom_y + np.random.random_sample() * (rows - bottom_y)
    top_y_right = top_y + np.random.random_sample() * (rows - top_y)
    if np.random.random_sample() <= 0.5:
        bottom_y_right = bottom_y - np.random.random_sample() * (bottom_y)
        top_y_right = top_y - np.random.random_sample() * (top_y)
    
    poly = np.asarray([[ [top_y,0], [bottom_y, cols], [bottom_y_right, cols], [top_y_right,0]]], dtype=np.int32)
        
    mask_weight = np.random.uniform(w_low, w_high)
    origin_weight = 1 - mask_weight
    
    mask = np.copy(img).astype(np.int32)
    cv2.fillPoly(mask, poly, (0, 0, 0))
    #masked_image = cv2.bitwise_and(img, mask)
    
    return cv2.addWeighted(img.astype(np.int32), origin_weight, mask, mask_weight, 0).astype(np.uint8)

def translate_image(img, mask, st_angle, translation_x, translation_y, delta_st_angle_per_px):
    """
    Shifts the image right, left, up or down. 
    When performing a lateral shift, a delta proportional to the pixel shifts is added to the current steering angle 
    """
    rows, cols = (img.shape[0], img.shape[1])
    #translation_x = np.random.randint(low_x_range, high_x_range) 
    #translation_y = np.random.randint(low_y_range, high_y_range) 
    
    st_angle += translation_x * delta_st_angle_per_px
    translation_matrix = np.float32([[1, 0, translation_x],[0, 1, translation_y]])
    img = cv2.warpAffine(img, translation_matrix, (cols, rows))
    mask = cv2.warpAffine(mask, translation_matrix, (cols, rows))
    
    return img, mask, st_angle

def augment_image(img, mask, translation_x, translation_y, p=0.33):
    """
    Augment a given image, by applying a series of transformations, with a probability p.
    The steering angle may also be modified.
    Returns the tuple (augmented_image, new_steering_angle)
    """
    aug_img = img
    aug_mask = mask
    angle_sign = 1
    angle_shift = 0
    
    if np.random.random_sample() <= 0.5: 
        aug_img, aug_mask = fliph_image(aug_img, aug_mask)
        angle_sign = -1
     
    if np.random.random_sample() <= 0.1:
        aug_img = change_image_brightness_rgb(aug_img)
    
    if np.random.random_sample() <= 0.1: 
        aug_img = add_random_shadow(aug_img, w_low=0.45)
            
    if np.random.random_sample() <= 0.25:
        aug_img, aug_mask, angle_shift = translate_image(aug_img, aug_mask, 0, translation_x, translation_y, 0.45*0.35/100.0)
            
    return aug_img, aug_mask, angle_sign, angle_shift


class SubsetSampler(Sampler):
    r"""Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class Drive360Loader(DataLoader):

    def __init__(self, config, phase, p_aug=0, road_idx=None):

        self.drive360 = Drive360(config, phase, p_aug, road_idx)
        batch_size = config['data_loader'][phase]['batch_size']
        sampler = SubsetSampler(self.drive360.indices)
        num_workers = config['data_loader'][phase]['num_workers']

        super().__init__(dataset=self.drive360,
                         batch_size=batch_size,
                         sampler=sampler,
                         num_workers=num_workers
                         )

class Drive360(object):
    ## takes a config json object that specifies training parameters and a
    ## phase (string) to specifiy either 'train', 'test', 'validation'
    def __init__(self, config, phase, p_aug=0, road_idx=None):
        self.road_idx = road_idx
        self.p_aug = p_aug
        self.config = config
        self.data_dir = config['data_loader']['data_dir']
        self.csv_name = config['data_loader'][phase]['csv_name']
        self.shuffle = config['data_loader'][phase]['shuffle']
        self.history_number = config['data_loader']['historic']['number']
        self.history_frequency = config['data_loader']['historic']['frequency']
        self.normalize_targets = config['target']['normalize']
        self.target_mean = {}
        target_mean = config['target']['mean']
        for k, v in target_mean.items():
            self.target_mean[k] = np.asarray(v, dtype=np.float32)
        self.target_std = {}
        target_std = config['target']['std']
        for k, v in target_std.items():
            self.target_std[k] = np.asarray(v, dtype=np.float32)

        self.front = self.config['front']
        self.right_left = config['multi_camera']['right_left']
        self.rear = config['multi_camera']['rear']

        #### reading in dataframe from csv #####
        self.dataframe = pd.read_csv(os.path.join(self.data_dir, self.csv_name),
                                     dtype={'cameraFront': object,
                                            'cameraRear': object,
                                            'cameraRight': object,
                                            'cameraLeft': object,
                                            'canSpeed': np.float32,
                                            'canSteering': np.float32
                                            })

        # Here we calculate the temporal offset for the starting indices of each chapter. As we cannot cross chapter
        # boundaries but would still like to obtain a temporal sequence of images, we cannot start at index 0 of each chapter
        # but rather at some index i such that the i-max_temporal_history = 0
        # To explain see the diagram below:
        #
        #             chapter 1    chapter 2     chapter 3
        #           |....-*****| |....-*****| |....-*****|
        # indices:   0123456789   0123456789   0123456789
        #
        # where . are ommitted indices and - is the index. This allows using [....] as temporal input.
        #
        # Thus the first sample will consist of images:     [....-]
        # Thus the second sample will consist of images:    [...-*]
        # Thus the third sample will consist of images:     [..-**]
        # Thus the fourth sample will consist of images:    [.-***]
        # Thus the fifth sample will consist of images:     [-****]
        # Thus the sixth sample will consist of images:     [*****]

        self.sequence_length = self.history_number*self.history_frequency
        max_temporal_history = self.sequence_length
        self.indices = self.dataframe.groupby('chapter').apply(lambda x: x.iloc[max_temporal_history:]).index.droplevel(level=0).tolist()

        #### phase specific manipulation #####
        if phase == 'train':
            self.dataframe['canSteering'] = np.clip(self.dataframe['canSteering'], a_max=360, a_min=-360)
            idx_2 = self.dataframe[(self.dataframe['canSteering']>10.) | (self.dataframe['canSteering']<-10.)].index.tolist()
            idx_3 = self.dataframe[(self.dataframe['canSteering']<10.) & (self.dataframe['canSteering']>-10.)].index.tolist()
            shuffle(idx_3)
            idx_4 = [idx_3[i] for i in range(len(idx_2)//2)]
            idx_final = idx_2 + idx_4
            self.indices = list(set(self.indices).intersection(idx_final))
            ##### If you want to use binning on angle #####
            ## START ##
            # self.dataframe['bin_canSteering'] = pd.cut(self.dataframe['canSteering'],
            #                                            bins=[-360, -20, 20, 360],
            #                                            labels=['left', 'straight', 'right'])
            # gp = self.dataframe.groupby('bin_canSteering')
            # min_group = min(gp.apply(lambda x: len(x)))
            # bin_indices = gp.apply(lambda x: x.sample(n=min_group)).index.droplevel(level=0).tolist()
            # self.indices = list(set(self.indices) & set(bin_indices))
            ## END ##

        elif phase == 'validation':
            self.dataframe['canSteering'] = np.clip(self.dataframe['canSteering'], a_max=360, a_min=-360)

        elif phase == 'test':
            # IMPORTANT: for the test phase indices will start 10s (100 samples) into each chapter
            # this is to allow challenge participants to experiment with different temporal settings of data input.
            # If challenge participants have a greater temporal length than 10s for each training sample, then they
            # must write a custom function here.

            # self.indices = self.dataframe.groupby('chapter').apply(
                # lambda x: x.iloc[100:]).index.droplevel(
                # level=0).tolist()
            if 'canSteering' not in self.dataframe.columns:
                self.dataframe['canSteering'] = [0.0 for _ in range(len(self.dataframe))]
            if 'canSpeed' not in self.dataframe.columns:
                self.dataframe['canSpeed'] = [0.0 for _ in range(len(self.dataframe))]


        if self.normalize_targets and not phase == 'test':
            self.dataframe['canSteering'] = (self.dataframe['canSteering'].values -
                                            self.target_mean['canSteering']) / self.target_std['canSteering']
            self.dataframe['canSpeed'] = (self.dataframe['canSpeed'].values -
                                            self.target_mean['canSpeed']) / self.target_std['canSpeed']

        if self.shuffle:
            shuffle(self.indices)



        print('Phase:', phase, '# of data:', len(self.indices))

        front_transforms = {
            'train': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=config['image']['norm']['mean'],
                                     std=config['image']['norm']['std'])
            ]),
            'validation': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=config['image']['norm']['mean'],
                                     std=config['image']['norm']['std'])
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=config['image']['norm']['mean'],
                                     std=config['image']['norm']['std'])
            ])}
        sides_transforms = {
            'train': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=config['image']['norm']['mean'],
                                     std=config['image']['norm']['std'])
            ]),
            'validation': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=config['image']['norm']['mean'],
                                     std=config['image']['norm']['std'])
            ]),
            'test': transforms.Compose([
                transforms.Resize((320, 180)),
                transforms.ToTensor(),
                transforms.Normalize(mean=config['image']['norm']['mean'],
                                     std=config['image']['norm']['std'])
            ])}

        self.imageFront_transform = front_transforms[phase]
        self.imageSides_transform = sides_transforms[phase]


    def __getitem__(self, index):
        inputs = {}
        labels = {}
        front_name = []
        end = index - self.sequence_length
        skip = int(-1 * self.history_frequency)
        rows = self.dataframe.iloc[index:end:skip].reset_index(drop=True, inplace=False)
        lower = -10.539250999999954
        higher = 0
        angle_sign = 1
        angle_shift = 0
        aug = random.random()
        brightness = np.random.uniform(0.2, 0.75)
        translation_x = np.random.randint(int(-60*0.45), int(61*0.45)) 
        translation_y = 0
        
        if self.front:
            inputs['cameraFront'] = {}
            inputs['cameraFront_mask'] = {}
            for row_idx, (_, row) in enumerate(rows.iterrows()):
                front_name.append(row['cameraFront'])
                img_front = Image.open(self.data_dir + row['cameraFront'])
                img_front_np = np.asarray(img_front)
                img_front_mask = Image.open('./Data/Sample1_mask/' + row['cameraFront'])
                img_front_mask_tmp = np.asarray(img_front_mask)
                
                if aug <= self.p_aug:
                    img_front_np, img_front_mask_tmp, angle_sign, angle_shift = augment_image(img_front_np, img_front_mask_tmp, translation_x, translation_y)
                    
                img_front_mask_tmp_2 = np.zeros_like(img_front_mask_tmp)
                for idx in list(self.road_idx.keys()):
                    img_front_mask_tmp_2[img_front_mask_tmp==idx] = self.road_idx[idx]

                inputs['cameraFront'][row_idx] = (self.imageFront_transform(img_front_np))               
                img_front_mask_tmp_2 = torch.nn.functional.one_hot(torch.tensor(img_front_mask_tmp_2).to(torch.int64), 20)
                img_front_mask_tmp_2 = img_front_mask_tmp_2.permute((2,0,1)).float()
                inputs['cameraFront_mask'][row_idx] = (img_front_mask_tmp_2)
        if self.right_left:
            inputs['cameraRight'] = self.imageSides_transform(Image.open(self.data_dir + rows['cameraRight'].iloc[0]))
            inputs['cameraLeft'] = self.imageSides_transform(Image.open(self.data_dir + rows['cameraLeft'].iloc[0]))
        if self.rear:
            inputs['cameraRear'] = self.imageSides_transform(Image.open(self.data_dir + rows['cameraRear'].iloc[0]))
        labels['canSteering'] = angle_sign*self.dataframe['canSteering'].iloc[index]+angle_shift
        labels['canSpeed'] = self.dataframe['canSpeed'].iloc[index]

        return inputs, labels, front_name