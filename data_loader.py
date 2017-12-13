import os
from os.path import join
import random
import numpy as np

class FileLoader(object):
	def __init__(self, filename, normalized=False):
		self.images = np.load(filename)	
		if normalized:
			self.images = self.images/255.0 * 2 - 1.

		self.current_index = 0
		self.num_images = len(self.images)
		self.image_shape = self.images[0].shape

	def get_batch(self, batch_size, randomize=False):
		if randomize:
			idx = random.sample(range(self.num_images), batch_size)
			return self.images[idx].reshape([-1] + list(self.images[0].shape))
		else:
			if self.current_index >= self.num_images:
				self.current_index = self.current_index % self.num_images
			if self.current_index + batch_size <= self.num_images:
				self.current_index += batch_size
				return self.images[self.current_index - batch_size:self.current_index].reshape([-1] + list(self.images[0].shape))
			else:
				first_part = self.images[self.current_index:].reshape([-1] + list(self.images[0].shape))
				self.current_index = (self.current_index + batch_size) % self.num_images
				second_part = self.images[:self.current_index].reshape([-1] + list(self.images[0].shape))
				return np.concatenate((first_part, second_part), 0).reshape([-1] + list(self.images[0].shape))
