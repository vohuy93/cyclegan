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
parser.add_argument('--testA', type=str, required=True, help='Path to a npy file containing test images of A dataset')
parser.add_argument('--testB', type=str, required=True, help='Path to a npy file containing test images of B dataset')
parser.add_argument('--folder', type=str, required=True)
parser.add_argument('--cp', type=int, required=True)


opt = parser.parse_args()
opt_dict = vars(opt)

trained_args = pickle.load(open(join(opt.folder, 'ops.pickle'), 'rb'))
for arg in trained_args:
    if arg not in ['folder', 'cp']:
        opt_dict[arg] = trained_args[arg]

print("Parameters of the model are: ...")
for arg in opt_dict:
	print(arg + ": " + str(opt_dict[arg]))


####################### READ DATA ####################################
testA = FileLoader(opt.testA, True)
testB = FileLoader(opt.testB, True)


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
	for i in range(10):
		image_A = testA.get_batch(1,True)
		image_B = testB.get_batch(1,True)
		outputB, outputA, outputB_cycle, outputA_cycle = sess.run([fake_B, fake_A, cycle_B, cycle_A], feed_dict={input_tensor_A:image_A, input_tensor_B:image_B})
		plt.imshow((image_A[0]+1.)/2)
		plt.show()
		plt.imshow((outputB[0]+1.)/2)
		plt.show()
		plt.imshow((outputA_cycle[0]+1.)/2)
		plt.show()
		plt.imshow((image_B[0]+1.)/2)
		plt.show()
		plt.imshow((outputA[0]+1.)/2)
		plt.show()
		plt.imshow((outputB_cycle[0]+1.)/2)
		plt.show()


