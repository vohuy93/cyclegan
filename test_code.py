import models
import tensorflow as tf
import numpy as np
import ops

#----------------------------------------------------------------------------------------------
# TEST tf.nn.moments

# sess = tf.Session()
# a = np.random.uniform(0.,50., (2,3,4,5))
# a = np.array(a, dtype=np.float32)

# mu, sigma2 = tf.nn.moments(tf.constant(a), [1,2], keep_dims=True)
# print(mu.get_shape().as_list())
# print(sigma2.get_shape().as_list())
# mu_ = tf.reduce_mean(tf.constant(a), [1,2], keep_dims=True)
# sigma2_ = tf.reduce_mean((tf.constant(a) - mu_)**2, [1,2], keep_dims=True)
# print(mu_.get_shape().as_list())
# print(sigma2_.get_shape().as_list())

# print(sess.run([tf.reduce_sum(tf.abs(mu - mu_)), tf.reduce_sum(tf.abs(sigma2_ - sigma2))]))

#----------------------------------------------------------------------------------------------

# Test tf.div

sess = tf.Session()
a = np.random.uniform(0.,5., (2,2))
b = np.random.uniform(0.,5., (2,2))
print(a)
print(b)
print(sess.run(tf.div(a,b)))
print(a/b)