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
import matplotlib.pyplot as plt

import argparse



####################### READ ARGUMENTS ###############################
parser = argparse.ArgumentParser()
parser.add_argument('--testA', type=str, required=True, help='Path to a folder containing test images of A dataset')
parser.add_argument('--testB', type=str, required=True, help='Path to a folder containing test images of B dataset')
parser.add_argument('--folder', type=str, required=True)
parser.add_argument('--cp', type=int, required=True)
parser.add_argument('--save_result', type=lambda x : x == 'True', default=False, help='Whether to save results')
parser.add_argument('--show_result', type=lambda x : x == 'True', default=True, help='Whether to save results')
parser.add_argument('--dest', type=str, default=None, help='folder to save fake and cycle')


opt = parser.parse_args()
opt_dict = vars(opt)

trained_args = pickle.load(open(join(opt.folder, 'ops.pickle'), 'rb'))
for arg in trained_args:
    if arg not in ['folder', 'cp']:
        opt_dict[arg] = trained_args[arg]

print("Parameters of the model are: ...")
for arg in opt_dict:
	print(arg + ": " + str(opt_dict[arg]))

if opt.save_result and opt.dest is None:
	print("Please provide dest to save results")
	exit(0)

if not os.path.exists(opt.dest):
	os.mkdir(opt.dest)
	os.mkdir(join(opt.dest, 'A'))
	os.mkdir(join(opt.dest, 'B'))
	os.mkdir(join(opt.dest, 'A', 'fake'))
	os.mkdir(join(opt.dest, 'A', 'cycle'))
	os.mkdir(join(opt.dest, 'B', 'fake'))
	os.mkdir(join(opt.dest, 'B', 'cycle'))


####################### READ DATA ####################################
# testA = FileLoader(opt.testA, True)
# testB = FileLoader(opt.testB, True)

filesA = [f for f in os.listdir(opt.testA) if f.endswith('.jpg')]
filesB = [f for f in os.listdir(opt.testB) if f.endswith('.jpg')]


####################### BUILD MODELS #################################

generator = eval('models.generator_%d'%opt.image_size)

input_tensor_A = tf.placeholder(tf.float32, [None, opt.image_size, opt.image_size, 3])
input_tensor_B = tf.placeholder(tf.float32, [None, opt.image_size, opt.image_size, 3])

fake_A = generator(input_tensor_B, 'generator_A', skip=opt.skip)
fake_B = generator(input_tensor_A, 'generator_B', skip=opt.skip)
cycle_A = generator(fake_B, 'generator_A', reuse=True, skip=opt.skip)
cycle_B = generator(fake_A, 'generator_B', reuse=True, skip=opt.skip)

saver = tf.train.Saver()
with tf.Session() as sess:
	saver.restore(sess, join(opt.folder, 'checkpoints/epoch_' + str(opt.cp) + '.cpkt'))
	for im in filesA:
		image_A = misc.imresize(ndimage.imread(join(opt.testA, im))[:,:,:3], (opt.image_size,opt.image_size)).reshape((1, opt.image_size, opt.image_size,3))/255.0 * 2.0 - 1.0

		outputB, outputA_cycle = sess.run([fake_B, cycle_A], feed_dict={input_tensor_A:image_A})
		if opt.show_result:
			plt.imshow((image_A[0]+1.)/2)
			plt.show()
			plt.imshow((outputB[0]+1.)/2)
			plt.show()
			plt.imshow((outputA_cycle[0]+1.)/2)
			plt.show()
		if opt.save_result:
			misc.imsave(join(opt.dest, 'B', 'fake', im), (outputB[0]+1.)/2)
			misc.imsave(join(opt.dest, 'B', 'cycle', im), (outputA_cycle[0]+1.)/2)

	for im in filesB:
		image_B = misc.imresize(ndimage.imread(join(opt.testB, im))[:,:,:3], (opt.image_size,opt.image_size)).reshape((1, opt.image_size, opt.image_size,3))/255.0 * 2.0 - 1.0
		outputA, outputB_cycle = sess.run([fake_A, cycle_B], feed_dict={input_tensor_B:image_B})	
		if opt.show_result:
			plt.imshow((image_B[0]+1.)/2)
			plt.show()
			plt.imshow((outputA[0]+1.)/2)
			plt.show()
			plt.imshow((outputB_cycle[0]+1.)/2)
			plt.show()
		if opt.save_result:
			misc.imsave(join(opt.dest, 'A', 'fake', im), (outputA[0]+1.)/2)
			misc.imsave(join(opt.dest, 'A', 'cycle', im), (outputB_cycle[0]+1.)/2)



