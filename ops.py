import tensorflow as tf
import numpy as np
from math import ceil

def reflect_padding(input, filter_size, stride):
    """
    An implementation of reflect padding for images before doing 
    convolution.

    Parameters:

        input (4-dimensional tensor): input
        filter_size (int): 2d size of square convolutional filters
        stride (int): stride 
    """
    in_height, in_width = input.get_shape().as_list()[1:3]

    out_height = ceil(in_height/stride)
    out_width  = ceil(in_width/stride)

    pad_along_height = max((out_height - 1) * stride + filter_size - in_height, 0)
    pad_along_width = max((out_width - 1) * stride + filter_size - in_width, 0)

    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    return tf.pad(input, [[0,0],[pad_top, pad_bottom],[pad_left, pad_right],[0,0]], 'REFLECT')

def conv2d(input, kernel_shape, scope_name, stride=1, stddev=0.02,
           const_bias=0.0, padding='REFLECT'):
    """
    Create a convolutional layer.

    Parameters:

        input (4-dimensional Tensor): input
        kernel_shape (list): shape of convolution filters of the form 
            [dim_width, dim_height, input_depth, output_depth]
        scope_name (string): name of scope of all variables created by the 
            function
        strides (int): (optional) 2d spatial stride
        stddev (float): (optional) standard deviation of random initialized 
            parameters
        const_bias (float): (optional) initial value of the bias term
        padding (string): (optional) type of padding method to use
    """
    with tf.variable_scope(scope_name):
        weights = tf.get_variable(name='weights', shape=kernel_shape,
                initializer=tf.truncated_normal_initializer(
                    mean=0.0, stddev=stddev)
                )
        biases = tf.get_variable(name='biases', shape=[kernel_shape[-1]],
                initializer=tf.constant_initializer(const_bias))
        if padding == 'REFLECT':
            input = reflect_padding(input, kernel_shape[0], stride)
            padding = 'VALID'
        conv = tf.nn.conv2d(input, weights, 
                            strides=[1,stride,stride,1], 
                            padding=padding) + biases
        return conv

def linear(input, out_dim, scope_name, stddev=0.02, const_bias=0.0):
    """
    Create a fully connected layer.

    Parameters:

        input (2-dimensional Tensor): input
        out_dim (int): number of units of the output
        scope_name (string): name of scope of all variables created by the 
            function
        stddev (float): (optional) standard deviation of random initialized 
            parameters
        const_bias (float): (optional) initial value of the bias term
    """
    weight_shape = [input.get_shape()[1], out_dim]
    with tf.variable_scope(scope_name):
        weights = tf.get_variable(name='weights', shape=weight_shape,
                initializer=tf.truncated_normal_initializer(
                    mean=0.0, stddev=stddev)
                )
        biases = tf.get_variable(name='biases', shape=[1],
                initializer=tf.constant_initializer(const_bias))
        fc = tf.matmul(input, weights) + biases
        return fc

def conv2d_trans(input, kernel_shape, output_shape, scope_name, 
                 stride=1, stddev=0.02, const_bias=0.0, padding='REFLECT'):
    """
    Create a transpose convolutional layer (deconvolutional layer).

    Parameters:
    
        input (4-dimensional Tensor): input
        kernel_shape (list): shape of convolution filters of the form 
            [dim_width, dim_height, output_depth, input_depth]
        output_shape: shape of output of the operator
        scope_name (string): name of scope of all variables created by the 
            function
        strides (int): (optional) 2d spatial stride
        stddev (float): (optional) standard deviation of random initialized 
            parameters
        const_bias (float): (optional) initial value of the bias term
        padding (string): (optional) type of padding method to use
    """
    with tf.variable_scope(scope_name):
        weights = tf.get_variable(name='weights', shape=kernel_shape,
                                  initializer=tf.truncated_normal_initializer(
                                    mean=0.0, stddev=stddev)
                                 )
        biases = tf.get_variable(name='biases', shape=[kernel_shape[-2]],
                 initializer=tf.constant_initializer(const_bias)
                 )
        if padding == 'REFLECT':
            input = reflect_padding(input, kernel_shape[0], stride)
            padding = 'VALID'
        conv = tf.nn.conv2d_transpose(input, weights,
                                      output_shape, strides=[1,stride,stride,1],
                                      padding=padding) + biases
        return conv

def lrelu(input, scope_name, leak=0.2):
    return tf.maximum(input, input*leak, name=scope_name)


def instance_norm(input, scope_name, epsilon=1e-8, stddev=0.02, const_bias=0.0):
    with tf.variable_scope(scope_name):
        num_channels = input.get_shape().as_list()[-1]
        mu = tf.reduce_mean(input, [1,2], keep_dims=True)
        sigma2 = tf.reduce_mean((input - mu)**2, [1,2], keep_dims=True)
        scale = tf.get_variable('scale', [num_channels],
                                initializer=tf.truncated_normal_initializer(
                                        mean=1.0, stddev=stddev
                                ))
        biases = tf.get_variable('biases', [num_channels], 
                               initializer=tf.constant_initializer(const_bias))
        return scale * tf.div(input - mu, tf.sqrt(sigma2 + epsilon)) + biases


def residual_block(input, kernel_shape, scope_name, stddev=0.02,
                   const_bias=0.0, padding='REFLECT'):
    """
    Create residual block.

    Parameters:

        input (4-dimensional Tensor): input
        kernel_shape (list): shape of convolution filters of the form 
            [dim_width, dim_height, input_depth, output_depth]
        scope_name (string): name of scope of all variables created by the 
            function
        strides (int): (optional) 2d spatial stride
        stddev (float): (optional) standard deviation of random initialized 
            parameters
        const_bias (float): (optional) initial value of the bias term
        padding (string): (optional) type of padding method to use
    """

    with tf.variable_scope(scope_name):
        conv1 = conv2d(input, kernel_shape, 'conv1', padding=padding)
        relu1 = tf.nn.relu(conv1, name='relu1')
        conv2 = conv2d(relu1, kernel_shape, 'conv2', padding=padding)
        return tf.nn.relu(conv2 + input, name='relu2')


def conv_instn_relu(input, kernel_shape, scope_name, stride=1, stddev=0.02,
                    const_bias=0.0, padding='REFLECT'):
    """
    Create a block convolution-instance_norm-relu.

    Parameters:

        input (4-dimensional Tensor): input
        kernel_shape (list): shape of convolution filters of the form 
            [dim_width, dim_height, input_depth, output_depth]
        scope_name (string): name of scope of all variables created by the 
            function
        strides (int): (optional) 2d spatial stride
        stddev (float): (optional) standard deviation of random initialized 
            parameters
        const_bias (float): (optional) initial value of the bias term
        padding (string): (optional) type of padding method to use
    """

    with tf.variable_scope(scope_name):
        conv = conv2d(input, kernel_shape, 'conv', stride, padding=padding)
        instn = instance_norm(conv, 'instance_norm')
        return tf.nn.relu(instn, name='relu')

def conv_instn_lrelu(input, kernel_shape, scope_name, stride=1, stddev=0.02,
                    const_bias=0.0, padding='REFLECT'):
    """
    Create a block convolution-instance_norm-lrelu.

    Parameters:

        input (4-dimensional Tensor): input
        kernel_shape (list): shape of convolution filters of the form 
            [dim_width, dim_height, input_depth, output_depth]
        scope_name (string): name of scope of all variables created by the 
            function
        strides (int): (optional) 2d spatial stride
        stddev (float): (optional) standard deviation of random initialized 
            parameters
        const_bias (float): (optional) initial value of the bias term
        padding (string): (optional) type of padding method to use
    """

    with tf.variable_scope(scope_name):
        conv = conv2d(input, kernel_shape, 'conv', stride, padding=padding)
        instn = instance_norm(conv, 'instance_norm')
        return lrelu(instn, scope_name='lrelu')

def conv_lrelu(input, kernel_shape, scope_name, stride=1, stddev=0.02,
                    const_bias=0.0, padding='REFLECT'):
    """
    Create a block convolution-instance_norm-lrelu.

    Parameters:

        input (4-dimensional Tensor): input
        kernel_shape (list): shape of convolution filters of the form 
            [dim_width, dim_height, input_depth, output_depth]
        scope_name (string): name of scope of all variables created by the 
            function
        strides (int): (optional) 2d spatial stride
        stddev (float): (optional) standard deviation of random initialized 
            parameters
        const_bias (float): (optional) initial value of the bias term
        padding (string): (optional) type of padding method to use
    """

    with tf.variable_scope(scope_name):
        conv = conv2d(input, kernel_shape, 'conv', stride, padding=padding)
        return lrelu(conv, scope_name='lrelu')

def convt_instn_relu(input, kernel_shape, output_shape, scope_name, 
                     stride=1, stddev=0.02, const_bias=0.0, padding='REFLECT'):
    """
    Create a block conv_transpose-instance_norm_relu.

    Parameters:
    
        input (4-dimensional Tensor): input
        kernel_shape (list): shape of convolution filters of the form 
            [dim_width, dim_height, output_depth, input_depth]
        output_shape: shape of output of the operator
        scope_name (string): name of scope of all variables created by the 
            function
        strides (int): (optional) 2d spatial stride
        stddev (float): (optional) standard deviation of random initialized 
            parameters
        const_bias (float): (optional) initial value of the bias term
        padding (string): (optional) type of padding method to use
    """

    with tf.variable_scope(scope_name):
        convt = conv2d_trans(input, kernel_shape, output_shape, 'convt', 
                                 stride, padding=padding)
        instn = instance_norm(convt, 'instance_norm')
        return tf.nn.relu(instn, name='relu')

def upsampling_conv_instn_relu(input, kernel_shape, output_shape, scope_name, 
                               stride=1, stddev=0.02, const_bias=0.0, 
                               padding='REFLECT'):
    """
    Create a block upsampling-conv_transpose-instance_norm_relu.

    Parameters:
    
        input (4-dimensional Tensor): input
        kernel_shape (list): shape of convolution filters of the form 
            [dim_width, dim_height, input_depth, output_depth]
        output_shape: shape of output of the operator
        scope_name (string): name of scope of all variables created by the 
            function
        strides (int): (optional) 2d spatial stride
        stddev (float): (optional) standard deviation of random initialized 
            parameters
        const_bias (float): (optional) initial value of the bias term
        padding (string): (optional) type of padding method to use
    """
    with tf.variable_scope(scope_name):
        upsampling = tf.image.resize(input, output_shape[1:3], 
                                     tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        conv = conv2d(upsampling, kernel_shape, 'conv', stride, padding=padding)
        instn = instance_norm(conv, 'instance_norm')
        return tf.nn.relu(instn, name='relu')





#####################################################################
# code by carpedm :                                                 #
# https://github.com/carpedm20/DCGAN-tensorflow/blob/master/ops.py  #
#####################################################################
class batch_norm(object):
  def __init__(self, name="batch_norm", epsilon=1e-5, momentum = 0.9):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum, 
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      center=True,
                      param_initializers={'beta':tf.zeros_initializer(), 
                      'gamma':tf.random_normal_initializer(1.0, 0.02)
                      },
                      is_training=train,
                      scope=self.name)

###########################################################################