from __future__ import print_function, division
import glob
import os 
import random
from skimage import io, transform, color
import numpy as np 
import torch
from torch.utils.data import Dataset
data_dir = os.path.join(os.getcwd(), 'data' + os.sep)
print(data_dir)
# Korean Name Card Datasets
class KNC_Dataset(Dataset):
    def __init__(self, img_list, mask_list, transform):
        self.img_list = img_list
        self.mask_list = mask_list
        self.transform = transform
        
    def __len__(self):
        return self.img_list
    
    def __getitem__(self, idx):
        img = io.imread(self.img_list[idx])
        img_name = self.img_list[idx]
        img_idx = np.array([idx])

        # Iter mask list
        if len(self.mask_list) == 0:
            mask_rbg = np.zeros(img.shape)
        else:
            mask_rbg = io.imread(self.mask_list[idx])
        
        mask = np.zeros(mask_rbg.shape[0:2])
        
        # Check Mask 3 channels
        if len(mask_rbg.shape) ==3:
            mask = mask_rbg[:,:,0]
        elif len(mask_rbg.shape) ==2:
            mask = mask_rbg

        # Assure img & mask has 3 channels 
        if len(img.shape)==3 and len(mask.shape)==2:
            mask = mask[:, :, np.newaxis]
            
        elif len(img.shape)==2 and len(mask.shape)==2:
            img = img[:, :, np.newaxis]
            mask = mask[:, :, np.newaxis]

        # Transform
        if self.transform:
            sample = self.transform(sample)

        return {'img_idx':img_idx, 'img':img, 'mask':mask}
    
class Rescale(object):
	'''
 	Rescale size with output quality defined
  	'''
	def __init__(self, output_size):
		assert isinstance(output_size, (int,tuple))
		self.output_size = output_size

	def __call__(self, sample):
		# Image index, Image and Mask 
		img_idx, img, mask = sample['img_idx'], sample['img'],sample['mask']

		h, w = img.shape[:2]

		if isinstance(self.output_size,int):
			if h > w:
				new_h, new_w = self.output_size * (h/w), self.output_size
			else:
				new_h, new_w = self.output_size, self.output_size * (w/h)
		else:
			new_h, new_w = self.output_size

		new_h, new_w = int(new_h), int(new_w)

		# Resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
		# img = transform.resize(image,(new_h,new_w),mode='constant')
		# lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)

		img = transform.resize(img, (self.output_size,self.output_size), mode='constant')
		mask = transform.resize(mask, (self.output_size,self.output_size), mode='constant', order=0, preserve_range=True)

		return {'img_idx':img_idx, 'img':img, 'mask':mask}

class RandomCrop(object):
	"""
	Data Augmentation random 0.5
	"""
	def __init__(self,output_size):
		assert isinstance(output_size, (int, tuple))
  
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size
   
	def __call__(self, sample):
		# Image index, Image and Mask 
		img_idx, img, mask = sample['img_idx'], sample['img'],sample['mask']

		if random.random() >= 0.5:
			image, mask = image[::-1], mask[::-1]

		h, w = image.shape[:2]
		new_h, new_w = self.output_size

		# Random range 
		top = np.random.randint(0, h - new_h)
		left = np.random.randint(0, w - new_w)

  		# Crop 		
		img = image[top: top + new_h, left: left + new_w]
		mask = mask[top: top + new_h, left: left + new_w]

		return {'img_idx':img_idx, 'img':img, 'mask':mask}

class ToTensorLab(object):
	"""Convert ndarrays in sample to Tensors."""
	def __init__(self,flag=0):
		self.flag = flag

	def __call__(self, sample):

		img_idx, img, mask =sample['img_idx'], sample['img'], sample['mask']

		tmpMask = np.zeros(mask.shape)

		# Mask Normalize
		if np.max(mask) >= 1e-6:
			mask = mask / np.max(mask)
   
		# Temporary Mask 
		tmpMask[:,:,0] = mask[:,:,0]
		tmpMask = mask.transpose((2, 0, 1))
  
		# Change Image color space 
		# RGB and Lab colors
		if self.flag == 2:
			tmpImg = np.zeros((img.shape[0],img.shape[1],6))
			tmpImgt = np.zeros((img.shape[0],img.shape[1],3))
   
			if img.shape[2]==1:
				tmpImgt[:,:,0] = img[:,:,0]
				tmpImgt[:,:,1] = img[:,:,0]
				tmpImgt[:,:,2] = img[:,:,0]
			else:
				tmpImgt = img
    
			tmpImgtl = color.rgb2lab(tmpImgt)

			# Nomalize image to range [0,1]
			tmpImg[:,:,0] = (tmpImgt[:,:,0] - np.min(tmpImgt[:,:,0])) / (np.max(tmpImgt[:,:,0]) - np.min(tmpImgt[:,:,0]))
			tmpImg[:,:,1] = (tmpImgt[:,:,1] - np.min(tmpImgt[:,:,1])) / (np.max(tmpImgt[:,:,1]) - np.min(tmpImgt[:,:,1]))
			tmpImg[:,:,2] = (tmpImgt[:,:,2] - np.min(tmpImgt[:,:,2])) / (np.max(tmpImgt[:,:,2]) - np.min(tmpImgt[:,:,2]))
			tmpImg[:,:,3] = (tmpImgtl[:,:,0] - np.min(tmpImgtl[:,:,0])) / (np.max(tmpImgtl[:,:,0]) - np.min(tmpImgtl[:,:,0]))
			tmpImg[:,:,4] = (tmpImgtl[:,:,1] - np.min(tmpImgtl[:,:,1])) / (np.max(tmpImgtl[:,:,1]) - np.min(tmpImgtl[:,:,1]))
			tmpImg[:,:,5] = (tmpImgtl[:,:,2] - np.min(tmpImgtl[:,:,2])) / (np.max(tmpImgtl[:,:,2]) - np.min(tmpImgtl[:,:,2]))

			# tmpImg = tmpImg / (np.max(tmpImg) - np.min(tmpImg))
			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0])) / np.std(tmpImg[:,:,0])
			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1])) / np.std(tmpImg[:,:,1])
			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2])) / np.std(tmpImg[:,:,2])
			tmpImg[:,:,3] = (tmpImg[:,:,3]-np.mean(tmpImg[:,:,3])) / np.std(tmpImg[:,:,3])
			tmpImg[:,:,4] = (tmpImg[:,:,4]-np.mean(tmpImg[:,:,4])) / np.std(tmpImg[:,:,4])
			tmpImg[:,:,5] = (tmpImg[:,:,5]-np.mean(tmpImg[:,:,5])) / np.std(tmpImg[:,:,5])

  		# Lab color
		elif self.flag == 1:
			tmpImg = np.zeros((img.shape[0],img.shape[1],3))

			if img.shape[2]==1:
				tmpImg[:,:,0] = img[:,:,0]
				tmpImg[:,:,1] = img[:,:,0]
				tmpImg[:,:,2] = img[:,:,0]
			else:
				tmpImg = img

			tmpImg = color.rgb2lab(tmpImg)

			# tmpImg = tmpImg / (np.max(tmpImg) - np.min(tmpImg))
			tmpImg[:,:,0] = (tmpImg[:,:,0] - np.min(tmpImg[:,:,0])) / (np.max(tmpImg[:,:,0]) - np.min(tmpImg[:,:,0]))
			tmpImg[:,:,1] = (tmpImg[:,:,1] - np.min(tmpImg[:,:,1])) / (np.max(tmpImg[:,:,1]) - np.min(tmpImg[:,:,1]))
			tmpImg[:,:,2] = (tmpImg[:,:,2] - np.min(tmpImg[:,:,2])) / (np.max(tmpImg[:,:,2]) - np.min(tmpImg[:,:,2]))

			tmpImg[:,:,0] = (tmpImg[:,:,0] - np.mean(tmpImg[:,:,0])) / np.std(tmpImg[:,:,0])
			tmpImg[:,:,1] = (tmpImg[:,:,1] - np.mean(tmpImg[:,:,1])) / np.std(tmpImg[:,:,1])
			tmpImg[:,:,2] = (tmpImg[:,:,2] - np.mean(tmpImg[:,:,2])) / np.std(tmpImg[:,:,2])

  		# RGB color
		else:
			tmpImg = np.zeros((img.shape[0],img.shape[1],3))
			img = img/np.max(img)
			if img.shape[2]==1:
				tmpImg[:,:,0] = (img[:,:,0]-0.485)/0.229
				tmpImg[:,:,1] = (img[:,:,0]-0.485)/0.229
				tmpImg[:,:,2] = (img[:,:,0]-0.485)/0.229
			else:
				tmpImg[:,:,0] = (img[:,:,0]-0.485)/0.229
				tmpImg[:,:,1] = (img[:,:,1]-0.456)/0.224
				tmpImg[:,:,2] = (img[:,:,2]-0.406)/0.225

		tmpImg = tmpImg.transpose((2, 0, 1))

		return {'img_idx':torch.from_numpy(img_idx), 'img': torch.from_numpy(tmpImg), 'label': torch.from_numpy(tmpMask)}
