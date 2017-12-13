import tensorflow as tf
import numpy as np
import scipy.misc as misc
import scipy.ndimage as ndimage
from data_loader import FileLoader
import models

import argparse



####################### READ ARGUMENTS ###############################
parser = argparse.ArgumentParser()
parser.add_argument('--image_size', type=int, required=True, help='Dimension of input images, either 128 or 256')
parser.add_argument('--trainA', type=str, required=True, help='Path to a npy file containing training images of A dataset')
parser.add_argument('--trainB', type=str, required=True, help='Path to a npy file containing training images of B dataset')
parser.add_argument('--batch_size', type=int, default=1, help='Batch_size')
parser.add_argument('--lambda_critic', type=float, required=True)
parser.add_argument('--lambda_cycle', type=float, required=True)
parser.add_argument('--num_epochs', type=int)
parser.add_argument('--num_iterations', type=int)
parser.add_argument('--n_critics', type=int)


opt = parser.parse_args()
opt_dict = vars(opt)

####################### READ DATA ####################################
trainA = FileLoader(opt.trainA, True)
trainB = FileLoader(opt.trainB, True)


####################### BUILD MODELS #################################

generator = eval('models.generator_%d'%opt.image_size)

input_tensor_A = tf.placeholder(tf.float32, [None, opt.image_size, opt.image_size, 3])
input_tensor_B = tf.placeholder(tf.float32, [None, opt.image_size, opt.image_size, 3])

fake_B = generator(input_tensor_A, 'generator_A')
fake_A = generator(input_tensor_B, 'generator_B')
cycle_A = generator(fake_B, 'generator_B', reuse=True)
cycle_B = generator(fake_A, 'generator_A', reuse=True)


# define discriminator_A's loss
epsilon_A = tf.random_uniform([opt.batch_size], maxval=1.0)
# print(epsilon_A.get_shape().as_list())
fake_A_bar = epsilon_A * input_tensor_A + (1 - epsilon_A) * fake_A
fake_A_critic = models.discriminator(fake_A_bar, 'discriminator_A')
# print(fake_A_critic.get_shape().as_list())
real_A_critic = models.discriminator(input_tensor_A, 'discriminator_A', reuse=True)
grad_A_critic = tf.gradients(tf.reduce_mean(fake_A_critic), fake_A_bar)
# print(len(grad_A_critic))
# print(grad_A_critic[0].get_shape().as_list())
grad_A_critic_penalty = tf.reduce_mean([(tf.norm(grad_A_critic[0][i]) - 1.0)**2 for i in range(opt.batch_size)])
# print(grad_A_critic_penalty.get_shape().as_list())
A_critic_loss = tf.reduce_mean(fake_A_critic - real_A_critic) + opt.lambda_critic * grad_A_critic_penalty
# print(A_critic_loss.get_shape().as_list())


# define discriminator_B's loss
epsilon_B = tf.random_uniform([opt.batch_size], maxval=1.0)
fake_B_bar = epsilon_B * input_tensor_B + (1 - epsilon_B) * fake_B
fake_B_critic = models.discriminator(fake_B_bar, 'discriminator_B')
real_B_critic = models.discriminator(input_tensor_B, 'discriminator_B', reuse=True)
grad_B_critic = tf.gradients(tf.reduce_mean(fake_B_critic), fake_B_bar)
grad_B_critic_penalty = tf.reduce_mean([(tf.norm(grad_B_critic[0][i]) - 1.0)**2 for i in range(opt.batch_size)])
B_critic_loss = tf.reduce_mean(fake_B_critic - real_B_critic) + opt.lambda_critic * grad_B_critic_penalty


# define losses for generators
cycle_loss = tf.reduce_mean(tf.abs(cycle_A - input_tensor_A)) + tf.reduce_mean(tf.abs(cycle_B - input_tensor_B))
A_gen_loss =  opt.lambda_cycle * cycle_loss - A_critic_loss
B_gen_loss = opt.lambda_cycle * cycle_loss - B_critic_loss


A_gen_var = [var for var in tf.global_variables() if 'generator_A' in var.name]
B_gen_var = [var for var in tf.global_variables() if 'generator_B' in var.name]
A_critic_var = [var for var in tf.global_variables() if 'discriminator_A' in var.name]
B_critic_var = [var for var in tf.global_variables() if 'discriminator_B' in var.name]


learning_rate = tf.placeholder(tf.float32, [])
A_gen_trainer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.9).minimize(A_gen_loss, var_list=A_gen_var)
B_gen_trainer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.9).minimize(B_gen_loss, var_list=B_gen_var)
A_critic_trainer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.9).minimize(A_critic_loss, var_list=A_critic_var)
B_critic_trainer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.9).minimize(B_critic_loss, var_list=B_critic_var)



####################### PREPARE FOLDER ###############################


####################### TRAIN NETWORK ################################

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for _epoch in range(opt.num_epochs):
	for _iteration in range(opt.num_iterations):
		# train critics
		for _ in range(opt.n_critics):
			_, A_critic_loss_val = sess.run([A_critic_trainer, A_critic_loss],
											feed_dict={input_tensor_A:trainA.get_batch(1,True),
													   input_tensor_B:trainB.get_batch(1,True),
													   learning_rate:0.0001})
			_, B_critic_loss_val = sess.run([B_critic_trainer, B_critic_loss],
										feed_dict={input_tensor_A:trainA.get_batch(1,True),
												   input_tensor_B:trainB.get_batch(1,True),
												   learning_rate:0.0001})
		_, A_gen_loss_val, A_critic_loss_val = sess.run([A_gen_trainer, A_gen_loss, A_critic_loss],
												feed_dict={input_tensor_A:trainA.get_batch(1),
														   input_tensor_B:trainB.get_batch(1),
														   learning_rate:0.0001})
		_, B_gen_loss_val, B_critic_loss_val = sess.run([B_gen_trainer, B_gen_loss, B_critic_loss],
												feed_dict={input_tensor_A:trainA.get_batch(1),
														   input_tensor_B:trainB.get_batch(1),
														   learning_rate:0.0001})

		print("A_gen_loss: %f ## B_gen_loss: %f ## A_critic_loss: %f ## B_critic_loss: %f\n"%(
				A_gen_loss_val, B_gen_loss_val, A_critic_loss_val, B_critic_loss_val))


