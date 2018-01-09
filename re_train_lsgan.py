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



############################################# READ ARGUMENTS ###############################################################
parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, required=True, help='Number of training epochs')
parser.add_argument('--folder', type=str, required=True)
parser.add_argument('--cp', type=int, required=True)

opt = parser.parse_args()
opt_dict = vars(opt)

trained_args = pickle.load(open(join(opt.folder, 'ops.pickle'), 'rb'))
for arg in trained_args:
    if arg not in ['folder', 'cp', 'num_epochs']:
        opt_dict[arg] = trained_args[arg]

############################################### READ DATA ##################################################################
trainA = FileLoader(opt.trainA, True)
trainB = FileLoader(opt.trainB, True)

if trainA.image_shape != (opt.image_size, opt.image_size, 3) or \
   trainB.image_shape != (opt.image_size, opt.image_size, 3):
   print("Data do not match the input image size")
   exit(0)

opt.num_iterations = max(trainA.num_images, trainB.num_images)


############################################## BUILD MODELS ################################################################

generator = eval('models.generator_%d'%opt.image_size)

input_tensor_A = tf.placeholder(tf.float32, [None, opt.image_size, opt.image_size, 3])
input_tensor_B = tf.placeholder(tf.float32, [None, opt.image_size, opt.image_size, 3])
fake_pool_A = tf.placeholder(tf.float32, [None, opt.image_size, opt.image_size, 3])
fake_pool_B = tf.placeholder(tf.float32, [None, opt.image_size, opt.image_size, 3])

fake_B = generator(input_tensor_A, 'generator_B', skip=opt.skip)
fake_A = generator(input_tensor_B, 'generator_A', skip=opt.skip)
cycle_A = generator(fake_B, 'generator_A', reuse=True, skip=opt.skip)
cycle_B = generator(fake_A, 'generator_B', reuse=True, skip=opt.skip)


# define discriminator_A's loss
fake_pool_A_dis = models.discriminator(fake_pool_A, 'discriminator_A')
fake_A_dis = models.discriminator(fake_A, 'discriminator_A', reuse=True)
real_A_dis = models.discriminator(input_tensor_A, 'discriminator_A', reuse=True)
A_dis_loss_real = tf.reduce_mean(tf.square(real_A_dis - 1))
A_dis_loss_fake_pool = tf.reduce_mean(tf.square(fake_pool_A_dis))
A_dis_loss = (A_dis_loss_fake_pool + A_dis_loss_real) * 0.5


# define discriminator_B's loss
fake_pool_B_dis = models.discriminator(fake_pool_B, 'discriminator_B')
fake_B_dis = models.discriminator(fake_B, 'discriminator_B', reuse=True)
real_B_dis = models.discriminator(input_tensor_B, 'discriminator_B', reuse=True)
B_dis_loss_real = tf.reduce_mean(tf.square(real_B_dis - 1))
B_dis_loss_fake_pool = tf.reduce_mean(tf.square(fake_pool_B_dis))
B_dis_loss = (B_dis_loss_fake_pool + B_dis_loss_real) * 0.5


# define losses for generators
cycle_loss = tf.reduce_mean(tf.abs(cycle_A - input_tensor_A)) + tf.reduce_mean(tf.abs(cycle_B - input_tensor_B))
A_gen_loss_fake = tf.reduce_mean(tf.square(fake_A_dis - 1))
B_gen_loss_fake = tf.reduce_mean(tf.square(fake_B_dis - 1))
A_gen_loss =  opt.lambda_cycle * cycle_loss + A_gen_loss_fake
B_gen_loss = opt.lambda_cycle * cycle_loss + B_gen_loss_fake



A_gen_var = [var for var in tf.global_variables() if 'generator_A' in var.name]
B_gen_var = [var for var in tf.global_variables() if 'generator_B' in var.name]
A_dis_var = [var for var in tf.global_variables() if 'discriminator_A' in var.name]
B_dis_var = [var for var in tf.global_variables() if 'discriminator_B' in var.name]


learning_rate = tf.placeholder(tf.float32, [])
if opt.optimizer == 'Adam':
    A_gen_trainer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999).minimize(A_gen_loss, var_list=A_gen_var)
    B_gen_trainer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999).minimize(B_gen_loss, var_list=B_gen_var)
    A_dis_trainer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999).minimize(A_dis_loss, var_list=A_dis_var)
    B_dis_trainer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999).minimize(B_dis_loss, var_list=B_dis_var)
else:
    A_gen_trainer = tf.train.RMSPropOptimizer(learning_rate).minimize(A_gen_loss, var_list=A_gen_var)
    B_gen_trainer = tf.train.RMSPropOptimizer(learning_rate).minimize(B_gen_loss, var_list=B_gen_var)
    A_dis_trainer = tf.train.RMSPropOptimizer(learning_rate).minimize(A_dis_loss, var_list=A_dis_var)
    B_dis_trainer = tf.train.RMSPropOptimizer(learning_rate).minimize(B_dis_loss, var_list=B_dis_var)


for var in tf.global_variables():
    print(var.name, ': ', var.get_shape().as_list())

saver = tf.train.Saver(max_to_keep=40)


########################################### PREPARE TRAINING FOLDER #######################################################
# create directories for containing checkpoints, constructed images during training and 
# log info for tensorboard

print("Preparing training directory...")

top_dir = join(opt.folder, utils.get_time())
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
real_writer = tf.summary.FileWriter(log_dir + '/real')
fake_writer = tf.summary.FileWriter(log_dir + '/fake')

######################### DEFINE SOME FUNCTIONS USED IN TRAINING #############################################

def write_summary(writer, val_dict, step):
    """
    Write log info to visualize with tensorboard
    """
    summaries = []
    for key in val_dict:
        summaries.append(tf.Summary.Value(tag=key, simple_value=val_dict[key]))
    writer.add_summary(tf.Summary(value=summaries), step)

def fake_pool_sample(fake_pool, fake_image, max_size):
    """
    A manager of fake pool of images: 
        - take a fake image from the history
        - add the new fake image to history
    """
    if len(fake_pool) < max_size:
        fake_pool.append(fake_image)
        return fake_image
    else:
        u = np.random.uniform(0,1)
        if u < 0.5:
            sample = np.random.randint(0, len(fake_pool))
            chosen_fake = fake_pool[sample]
            fake_pool[sample] = fake_image
            return chosen_fake
        else:
            return fake_image


####################################### TRAIN NETWORK ##########################################################

# create a session and initialize all variables
sess = tf.Session()
# sess.run(tf.global_variables_initializer())
saver.restore(sess, join(opt.folder, 'checkpoints', 'epoch_%d.cpkt'%opt.cp))

# create a list of epochs where a checkpoint will be saved
time_to_save = list(range(1,5)) + list(range(6, 20, 2)) + list(range(20, 100, 5)) + list(range(100, 1001, 10))
begin_time = time.time()

# pools of fake images for discriminator A and discriminator B
pool_A = []
pool_B = []

# begin training
for _epoch in range(opt.cp+1, opt.num_epochs):
    # compute learning rate for each epoch
    if _epoch < 100:
        current_lr = opt.learning_rate
    else:
        current_lr = opt.learning_rate - opt.learning_rate * (_epoch - 100)/(opt.num_epochs - 100)


    for _iteration in range(opt.num_iterations):
        # train generators
        _, _, fake_A_val, fake_B_val, \
        A_gen_loss_val, B_gen_loss_val, \
        A_gen_loss_fake_val, B_gen_loss_fake_val = sess.run(
                                                        [
                                                            A_gen_trainer, B_gen_trainer, 
                                                            fake_A, fake_B, 
                                                            A_gen_loss, B_gen_loss,
                                                            A_gen_loss_fake, B_gen_loss_fake
                                                        ], 
                                                        feed_dict={
                                                            input_tensor_A:trainA.get_batch(1),
                                                            input_tensor_B:trainB.get_batch(1),
                                                            learning_rate:current_lr
                                                        }
                                                    )
    
        # train discriminators
        _, _, A_dis_loss_val, B_dis_loss_val, \
        A_dis_loss_real_val, B_dis_loss_real_val, \
        A_dis_loss_fake_pool_val, B_dis_loss_fake_pool_val = sess.run(
                                                    [
                                                        A_dis_trainer, B_dis_trainer,
                                                        A_dis_loss, B_dis_loss,
                                                        A_dis_loss_real, B_dis_loss_real,
                                                        A_dis_loss_fake_pool, B_dis_loss_fake_pool
                                                    ],
                                                    feed_dict={
                                                        input_tensor_A:trainA.get_batch(1,True),
                                                        input_tensor_B:trainB.get_batch(1, True),
                                                        fake_pool_A:fake_pool_sample(pool_A, fake_A_val, opt.fake_pool_size),
                                                        fake_pool_B:fake_pool_sample(pool_B, fake_B_val, opt.fake_pool_size),
                                                        learning_rate:current_lr
                                                    }
                                                )



        write_summary(summary_writer, 
            {'A_gen_loss':A_gen_loss_val, 'B_gen_loss':B_gen_loss_val, 'A_dis_loss':A_dis_loss_val, 'B_dis_loss':B_dis_loss_val}, 
            _iteration + _epoch * opt.num_iterations + 1)
        write_summary(real_writer,
            {'A_dis_loss':A_dis_loss_real_val, 'B_dis_loss':B_dis_loss_real_val},
            _iteration + _epoch * opt.num_iterations + 1)
        write_summary(fake_writer,
            {'A_dis_loss':A_dis_loss_fake_pool_val, 'B_dis_loss':B_dis_loss_fake_pool_val, 
             'A_gen_loss':A_gen_loss_fake_val, 'B_gen_loss':B_gen_loss_fake_val},
            _iteration + _epoch * opt.num_iterations + 1)

        print("A_gen_loss: %f ## B_gen_loss: %f ## A_dis_loss: %f ## B_dis_loss: %f\n"%(
                A_gen_loss_val, B_gen_loss_val, A_dis_loss_val, B_dis_loss_val))
        print("Time is %f"%(time.time() - begin_time))
    if _epoch+1 in time_to_save:
        saver.save(sess, join(checkpoints_dir, 'epoch_%d.cpkt'%(_epoch+1)))


