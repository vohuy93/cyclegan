import tensorflow as tf
import ops
import numpy as np

def generator_128(input, scope_name, reuse=False, skip=False):
	"""
	Funtion to build the generator of a GAN.

	Parameters:

		
	"""
	with tf.variable_scope(scope_name, reuse=reuse):
		c7s1_32 = ops.conv_instn_relu(input, [7,7,3,32], 'c7s1-32', 1, padding="REFLECT")
		d64 = ops.conv_instn_relu(c7s1_32, [3,3,32,64], 'd64', 2, padding="SAME")
		d128 = ops.conv_instn_relu(d64, [3,3,64,128], 'd128', 2, padding="SAME")
		
		r128_1 = ops.residual_block(d128, [3,3,128,128], 'r128_1', padding="REFLECT")
		r128_2 = ops.residual_block(r128_1, [3,3,128,128], 'r128_2', padding="REFLECT")
		r128_3 = ops.residual_block(r128_2, [3,3,128,128], 'r128_3', padding="REFLECT")
		r128_4 = ops.residual_block(r128_3, [3,3,128,128], 'r128_4', padding="REFLECT")
		r128_5 = ops.residual_block(r128_4, [3,3,128,128], 'r128_5', padding="REFLECT")
		r128_6 = ops.residual_block(r128_5, [3,3,128,128], 'r128_6', padding="REFLECT")

		u64 = ops.convt_instn_relu(r128_6, [3,3,64,128], tf.shape(d64), 'u64', 2, padding="SAME")
		u64.set_shape(d64.get_shape().as_list())
		u32 = ops.convt_instn_relu(u64, [3,3,32,64], tf.shape(c7s1_32), 'u32', 2, padding="SAME")
		u32.set_shape(c7s1_32.get_shape().as_list())
		c7s1_3 = ops.conv2d(u32, [7,7,32,3], 'c7s1-3', 1, padding="REFLECT")
		if skip:
			return tf.nn.tanh(c7s1_3 + input, "output")
		else:
			return tf.nn.tanh(c7s1_3, "output")



def generator_256(input, scope_name, reuse=False, skip=False):
	with tf.variable_scope(scope_name, reuse=reuse):
		c7s1_32 = ops.conv_instn_relu(input, [7,7,3,32], 'c7s1-32', 1, padding="REFLECT")
		d64 = ops.conv_instn_relu(c7s1_32, [3,3,32,64], 'd64', 2, padding="SAME")
		d128 = ops.conv_instn_relu(d64, [3,3,64,128], 'd128', 2, padding="SAME")

		r128_1 = ops.residual_block(d128, [3,3,128,128], 'r128_1', padding="REFLECT")
		r128_2 = ops.residual_block(r128_1, [3,3,128,128], 'r128_2', padding="REFLECT")
		r128_3 = ops.residual_block(r128_2, [3,3,128,128], 'r128_3', padding="REFLECT")
		r128_4 = ops.residual_block(r128_3, [3,3,128,128], 'r128_4', padding="REFLECT")
		r128_5 = ops.residual_block(r128_4, [3,3,128,128], 'r128_5', padding="REFLECT")
		r128_6 = ops.residual_block(r128_5, [3,3,128,128], 'r128_6', padding="REFLECT")
		r128_7 = ops.residual_block(r128_6, [3,3,128,128], 'r128_7', padding="REFLECT")
		r128_8 = ops.residual_block(r128_7, [3,3,128,128], 'r128_8', padding="REFLECT")
		r128_9 = ops.residual_block(r128_8, [3,3,128,128], 'r128_9', padding="REFLECT")

		u64 = ops.convt_instn_relu(r128_9, [3,3,64,128], tf.shape(d64), 'u64', 2, padding="SAME")
		u64.set_shape(d64.get_shape().as_list())
		u32 = ops.convt_instn_relu(u64, [3,3,32,64], tf.shape(c7s1_32), 'u32', 2, padding="SAME")
		u32.set_shape(c7s1_32.get_shape().as_list())
		c7s1_3 = ops.conv2d(u32, [7,7,32,3], 'c7s1-3', 1, padding="REFLECT")

		if skip:
			return tf.nn.tanh(c7s1_3 + input, "output")
		else:
			return tf.nn.tanh(c7s1_3, "output")


def discriminator(input, scope_name, reuse=False):
	with tf.variable_scope(scope_name, reuse=reuse):
<<<<<<< Updated upstream
		c64 = ops.lrelu(ops.conv2d(input, [4,4,3,64], 'c64/conv', 2, padding="SAME"), 'c64/lrelu')
		c128 = ops.conv_instn_lrelu(c64, [4,4,64,128], 'c128', 2, padding="SAME")
		c256 = ops.conv_instn_lrelu(c128, [4,4,128,256], 'c256', 2, padding="SAME")
		c512 = ops.conv_instn_lrelu(c256, [4,4,256,512], 'c512', 1, padding="SAME")
=======
		c64 = ops.lrelu(ops.conv2d(input, [4,4,3,64], 'c64/conv', 2, 
									padding="SAME"), 'c64/lrelu')
		c128 = ops.conv_instn_lrelu(c64, [4,4,64,128], 'c128', 2, 
									padding="SAME")
		c256 = ops.conv_instn_lrelu(c128, [4,4,128,256], 'c256', 2,
									padding="SAME")
		c512 = ops.conv_instn_lrelu(c256, [4,4,256,512], 'c512', 1,
									padding="SAME")
>>>>>>> Stashed changes
		return ops.conv2d(c512, [4,4,512,1], 'output', 1, padding="SAME")

def discriminator_improved_wgan(input, scope_name, reuse=False):
	with tf.variable_scope(scope_name, reuse=reuse):
		c64 = ops.lrelu(ops.conv2d(input, [4,4,3,64], 'c64/conv', 2, 
									padding="SAME"), 'c64/lrelu')
		c128 = ops.conv_lrelu(c64, [4,4,64,128], 'c128', 2, 
									padding="SAME")
		c256_1 = ops.conv_lrelu(c128, [4,4,128,256], 'c256_1', 2,
									padding="SAME")
		c256_2 = ops.conv_lrelu(c256_1, [4,4,256,256], 'c256_2', 2,
									padding="SAME")
		c512 = ops.conv_lrelu(c256_2, [4,4,256,512], 'c512', 2,
									padding="SAME")
		c512_shape = c512.get_shape().as_list()
		c_reshape = tf.reshape(c512, [-1, c512_shape[1] * c512_shape[2] * c512_shape[3]])
		return tf.nn.sigmoid(ops.linear(c_reshape, 1, 'output'))
		

