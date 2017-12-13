import numpy as np
import scipy.ndimage as ndimage
import scipy.misc as misc
from os.path import join
import os

def is_image(filename):
	return filename.endswith('jpg') or \
		   filename.endswith('JPG') or \
		   filename.endswith('JPEG') or \
		   filename.endswith('jpeg') or \
		   filename.endswith('png') or \
		   filename.endswith('PNG')

def images_to_array(source_folder, save_name, resize=None):
	files = [file for file in os.listdir(source_folder)]
	assert np.all([is_image(file) for file in files])

	images = []
	for _file in files:
		im = ndimage.imread(join(source_folder, _file))
		if len(im.shape) == 3 and im.shape[-1] >= 3:
			if resize is not None:
				im = misc.imresize(im, resize)
			images.append(im[:,:,:3])
	np.save(save_name, np.array(images))


"""
import preprocessing
import preprocessing as p
p.images_to_array('datasets/apple2orange/trainA', 'datasets/apple2orange/trainA_128.npy', (128,128))
p.images_to_array('datasets/apple2orange/trainB', 'datasets/apple2orange/trainB_128.npy', (128,128))
p.images_to_array('datasets/apple2orange/testB', 'datasets/apple2orange/testB_128.npy', (128,128))
p.images_to_array('datasets/apple2orange/testA', 'datasets/apple2orange/testA_128.npy', (128,128))

p.images_to_array('datasets/horse2zebra/trainA', 'datasets/horse2zebra/trainA_128.npy', (128,128))
p.images_to_array('datasets/horse2zebra/trainB', 'datasets/horse2zebra/trainB_128.npy', (128,128))
p.images_to_array('datasets/horse2zebra/testA', 'datasets/horse2zebra/testA_128.npy', (128,128))
p.images_to_array('datasets/horse2zebra/testB', 'datasets/horse2zebra/testB_128.npy', (128,128))

"""