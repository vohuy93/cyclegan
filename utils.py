from __future__ import division
import abc
import tensorflow as tf
import numpy as np 
import datetime
import matplotlib.pyplot as plt


def crop_center(image, fill='mean'):
    """
    Create a hole in an image. Dimensions of the hole are about half of
    dimensions of the image.
    """

    image_width, image_height = image.shape
    hole_width = image_width - 2 * (image_width + 3)//4 
    hole_height = image_height - 2 * (image_height + 3)//4
    hole_width_region = slice((image_width + 3)//4, 
                              (image_width + 3)//4  + hole_width) 
    hole_height_region = slice((image_height + 3)//4,
                               (image_height + 3)//4 + hole_height)

    # create a new image to store the cropped image
    new_image = np.array(image, copy=True)
    # cropping
    if fill == 'zero':
        new_image[hole_width_region, hole_height_region] = 0
    else:
        new_image[hole_width_region, hole_height_region] = \
                        np.mean(image[hole_width_region, hole_height_region])
    return new_image


def linear_normalization(data, min_val, max_val):
    # take min and max values in data
    min_data = np.min(data)
    max_data = np.max(data)
    # make sure that there are at least two distinct elements in data
    assert min_data < max_data
    # rescale data
    return min_val + (data-min_data)/(max_data-min_data)*(max_val-min_val) 
    

def show_image(image, gray_scale=False):
    if gray_scale:
        plt.gray()
    plt.imshow(image)
    plt.show()

def save_image(image, name, gray_scale=False):
    if gray_scale:
        plt.gray()
    plt.imsave(name, image)
    
def combine_images(images, arrangement='line',
                   width=None, height=None):
    """
    Parameters:
    
    arragement: 'line' or 'square' or 'rec'
    """
    if len(images.shape) != 3:
        print("Invalid input! You have to provide an array of images")
        exit(0)
    if arrangement == 'line':
        return np.concatenate(images, axis=1)
    elif arrangement == 'square':
        num_images = len(images)
        if int(np.sqrt(num_images))**2 != num_images:
            print("Number of images is %d"%num_images)
            print("Number of images has to be a square number")
            print("The combination could not be done")
            exit(0)
        width = int(np.sqrt(num_images))
        return np.concatenate(
                    [
                      np.concatenate(
                            images[i*width:i*width+width], 
                            axis=1
                            ) 
                      for i in range(width)
                    ], 
                    axis=0
                    )
    elif arrangement == 'rec':
        if width is None or height is None:
            print("Values of width and height are not given")
            exit(0)
        if width * height != len(images):
            print("Invalid values of width and height")
            print("Product of width and height should be equal to len(images)")
            exit(0)
        return np.concatenate(
                    [
                      np.concatenate(
                            images[i*width:i*width+width], 
                            axis=1
                            ) 
                      for i in range(height)
                    ], 
                    axis=0
                    )
            

def get_time():
    now = datetime.datetime.now()
    return '{}-{}-{}_{}-{}-{}'.format(
                now.year, now.month, now.day, now.hour, now.minute, now.second)
                
##############################################################################
# some implementations of sampler classes and functions                      #
# used to sample noises in GAN                                               #
##############################################################################

class NoiseSampler(object):
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def sample(self, num_images):
        pass
    @abc.abstractmethod
    def get_shape(self):
        pass

def uniform_noise_sampler(min_val, max_val, shape):
    """
    Return a sampler used to sample uniformly noise images. 
    """
    class Sampler(NoiseSampler):
        def __init__(self, low, high, shape):
            self.low = low
            self.high = high
            self.shape = shape
        def sample(self, num_images):
            return np.random.uniform(self.low, self.high, 
                                     [num_images] + list(self.shape))
        def get_shape(self):
            return self.shape
    return Sampler(min_val, max_val, shape)

def truncated_gaussian_noise_sampler(min_val, max_val, stddev, shape):
    """
    Return a sampler used to sample truncated gaussian noise images.
    """
    class Sampler(object):
        def __init__(self, a, b, scale, shape):
            self.min_val = a
            self.max_val = b
            self.stddev = scale
            self.shape = shape
        def sample(self, num_images):
            from scipy.stats import truncnorm
            data_shape = [num_images] + list(self.shape)
            return truncnorm(a=self.min_val, b=self.max_val, 
                             scale=self.stddev).rvs(size=data_shape)
        def get_shape(self):
            return self.shape
    return Sampler(min_val, max_val, stddev, shape)
