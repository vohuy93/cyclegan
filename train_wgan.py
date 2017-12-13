import tensorflow as tf
import numpy as np
import scipy.misc as misc
import scipy.ndimage as ndimage
from data_loader import FileLoader
import models
import shutil
import os
from os.path import join
import utils
import sys
import inspect
import pickle
import time

import argparse



####################### READ ARGUMENTS ###############################
parser = argparse.ArgumentParser()
parser.add_argument('--image_size', type=int, required=True, help='Dimension of input images, either 128 or 256')
parser.add_argument('--trainA', type=str, required=True, help='Path to a npy file containing training images of A dataset')
parser.add_argument('--trainB', type=str, required=True, help='Path to a npy file containing training images of B dataset')
parser.add_argument('--batch_size', type=int, default=1, help='Batch_size')
parser.add_argument('--lambda_cycle', type=float, default=10.0, help='Weight of cycle loss compared to discriminator loss')
parser.add_argument('--num_epochs', type=int, required=True, help='Number of training epochs')
parser.add_argument('--n_critics', type=int, default=5, help='Number of discriminator updates in each iteration')
parser.add_argument('--top_dir', type=str, required=True, help='The top folder in which training infos are saved')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--skip', type=lambda x: x == 'True', default=False, help='Whether to use a skip connection in generators')


opt = parser.parse_args()
opt_dict = vars(opt)

####################### READ DATA ####################################
trainA = FileLoader(opt.trainA, True)
trainB = FileLoader(opt.trainB, True)

if trainA.image_shape != (opt.image_size, opt.image_size, 3) or \
   trainB.image_shape != (opt.image_size, opt.image_size, 3):
   print("Data do not match the input image size")
   exit(0)

opt.num_iterations = max(trainA.num_images, trainB.num_images)


####################### BUILD MODELS #################################

generator = eval('models.generator_%d'%opt.image_size)

input_tensor_A = tf.placeholder(tf.float32, [None, opt.image_size, opt.image_size, 3])
input_tensor_B = tf.placeholder(tf.float32, [None, opt.image_size, opt.image_size, 3])

fake_B = generator(input_tensor_A, 'generator_B', skip=opt.skip)
fake_A = generator(input_tensor_B, 'generator_A', skip=opt.skip)
cycle_A = generator(fake_B, 'generator_A', reuse=True, skip=opt.skip)
cycle_B = generator(fake_A, 'generator_B', reuse=True, skip=opt.skip)


# define discriminator_A's loss
fake_A_critic = models.discriminator(fake_A, 'discriminator_A')
real_A_critic = models.discriminator(input_tensor_A, 'discriminator_A', reuse=True)
A_critic_loss = tf.reduce_mean(fake_A_critic - real_A_critic)
# print(A_critic_loss.get_shape().as_list())
# exit(0)


# define discriminator_B's loss
fake_B_critic = models.discriminator(fake_B, 'discriminator_B')
real_B_critic = models.discriminator(input_tensor_B, 'discriminator_B', reuse=True)
B_critic_loss = tf.reduce_mean(fake_B_critic - real_B_critic)


# define losses for generators
cycle_loss = tf.reduce_mean(tf.abs(cycle_A - input_tensor_A)) + tf.reduce_mean(tf.abs(cycle_B - input_tensor_B))
A_gen_loss =  opt.lambda_cycle * cycle_loss - tf.reduce_mean(fake_A_critic)
B_gen_loss = opt.lambda_cycle * cycle_loss - tf.reduce_mean(fake_B_critic)


A_gen_var = [var for var in tf.global_variables() if 'generator_A' in var.name]
B_gen_var = [var for var in tf.global_variables() if 'generator_B' in var.name]
A_critic_var = [var for var in tf.global_variables() if 'discriminator_A' in var.name]
B_critic_var = [var for var in tf.global_variables() if 'discriminator_B' in var.name]

A_critic_clip = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in A_critic_var]
B_critic_clip = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in B_critic_var]



learning_rate = tf.placeholder(tf.float32, [])
A_gen_trainer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999).minimize(A_gen_loss, var_list=A_gen_var)
B_gen_trainer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999).minimize(B_gen_loss, var_list=B_gen_var)
A_critic_trainer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999).minimize(A_critic_loss, var_list=A_critic_var)
B_critic_trainer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999).minimize(B_critic_loss, var_list=B_critic_var)


for var in tf.global_variables():
	print(var.name, ': ', var.get_shape().as_list())

saver = tf.train.Saver(max_to_keep=40)


####################### PREPARE FOLDER ###############################

print("Preparing training directory...")
# create directories for containing checkpoints, constructed images during training and 
# log info for tensorboard

top_dir = join(opt.top_dir, utils.get_time())
checkpoints_dir = join(top_dir, 'checkpoints')
images_dir = join(top_dir, 'images')
log_dir = join(top_dir, 'log')
config_file = join(top_dir, 'config_info.txt')

os.mkdir(top_dir)
os.mkdir(checkpoints_dir)
os.mkdir(images_dir)
os.mkdir(log_dir)

# write config info and copy command line to the file
with open(config_file,'w') as file:
    for key in sorted(opt_dict.keys()):
        file.write(key + ": " + str(opt_dict[key]) + "\n")
    file.write('python ')
    for arg in sys.argv:
        file.write(arg + " ")
    file.write('\n')

# copy source codes to the training folder
# this facillitates code version management
import shutil
source_file = os.path.realpath(__file__)
if source_file[-1] == 'c':
    source_file = source_file[:-1]
shutil.copy2(source_file, join(top_dir, 'source.py'))

models_path = inspect.getfile(models)
if models_path[-1] == 'c':
    models_path = models_path[:-1]
shutil.copy2(models_path, join(top_dir, 'models.py'))

# save config info to a .pickle file, this facillitates
# later training
with open(join(top_dir, 'ops.pickle'), 'wb') as file:
    pickle.dump(opt_dict, file, pickle.HIGHEST_PROTOCOL)


# create summary writer
summary_writer = tf.summary.FileWriter(log_dir + '/summary')

######################################################################

def write_summary(writer, val_dict, step):
    summaries = []
    for key in val_dict:
        summaries.append(tf.Summary.Value(tag=key, simple_value=val_dict[key]))
    writer.add_summary(tf.Summary(value=summaries), step)

####################### TRAIN NETWORK ################################

sess = tf.Session()
sess.run(tf.global_variables_initializer())

time_to_save = list(range(1,5)) + list(range(6, 20, 2)) + list(range(20, 100, 5)) + list(range(100, 1001, 10))
begin_time = time.time()
for _epoch in range(opt.num_epochs):
	if _epoch < 100:
		current_lr = opt.learning_rate
	else:
		current_lr = opt.learning_rate - opt.learning_rate * (_epoch - 100)/100
	for _iteration in range(opt.num_iterations):
		# train critics
		for _n_critics in range(opt.n_critics):
			_, A_critic_loss_val = sess.run([A_critic_trainer, A_critic_loss],
											feed_dict={input_tensor_A:trainA.get_batch(1,True),
													   input_tensor_B:trainB.get_batch(1,True),
													   learning_rate:current_lr})
			sess.run(A_critic_clip)
			_, B_critic_loss_val = sess.run([B_critic_trainer, B_critic_loss],
										feed_dict={input_tensor_A:trainA.get_batch(1,True),
												   input_tensor_B:trainB.get_batch(1,True),
												   learning_rate:current_lr})
			sess.run(B_critic_clip)

		_, A_gen_loss_val, A_critic_loss_val, _, B_gen_loss_val, B_critic_loss_val = \
											   sess.run([A_gen_trainer, A_gen_loss, A_critic_loss,
														 B_gen_trainer, B_gen_loss, B_critic_loss],
												feed_dict={input_tensor_A:trainA.get_batch(1),
														   input_tensor_B:trainB.get_batch(1),
														   learning_rate:current_lr})


		write_summary(summary_writer, 
			{'A_gen_loss':A_gen_loss_val, 'B_gen_loss':B_gen_loss_val, 'A_critic_loss':A_critic_loss_val, 'B_critic_loss':B_critic_loss_val}, 
			_iteration + _epoch * opt.num_iterations + 1)

		print("A_gen_loss: %f ## B_gen_loss: %f ## A_critic_loss: %f ## B_critic_loss: %f\n"%(
				A_gen_loss_val, B_gen_loss_val, A_critic_loss_val, B_critic_loss_val))
		print("Time is %f"%(time.time() - begin_time))
	if _epoch+1 in time_to_save:
		saver.save(sess, join(checkpoints_dir, 'epoch_%d.cpkt'%(_epoch+1)))


