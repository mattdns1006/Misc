import tensorflow as tf
import numpy as np
from importlib.machinery import SourceFileLoader
tfFns = SourceFileLoader("tfFns.py", "/Users/matt/misc/python/tfFns.py").load_module()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

batchSize = 10
x = tf.placeholder(tf.float32, shape=[None, 784])

#Encoding layers
x_image = tf.reshape(x, [batchSize,28,28,1])

feats = 8
W_conv1 = tfFns.weight_variable([3, 3, 1, 8])
b_conv1 = tfFns.bias_variable([8])

W_conv2 = tfFns.weight_variable([3, 3, feats, feats])
b_conv2 = tfFns.bias_variable([feats])

W_conv3 = tfFns.weight_variable([3, 3, feats, feats])
b_conv3 = tfFns.bias_variable([feats])

W_conv4 = tfFns.weight_variable([3, 3, feats, feats])
b_conv4 = tfFns.bias_variable([feats])

#Decoding layers
outputShape5 = [batchSize,8,8,feats]
W_deconv1 = tfFns.weight_variable([3, 3, feats, feats])
b_deconv1 = tfFns.bias_variable([feats])

outputShape6 = [batchSize,16,16,feats]
W_deconv2 = tfFns.weight_variable([3, 3, feats, feats])
b_deconv2 = tfFns.bias_variable([feats])

outputShape7 = [batchSize,28,28,1]
W_deconv3 = tfFns.weight_variable([3, 3, 1, feats])
b_deconv3 = tfFns.bias_variable([1])


# In[32]:

h_conv1 = tf.nn.relu(tfFns.conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = tfFns.max_pool_2x2(h_conv1)

h_conv2 = tf.nn.relu(tfFns.conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = tfFns.max_pool_2x2(h_conv2)

h_conv3 = tf.nn.relu(tfFns.conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = tfFns.max_pool_2x2(h_conv3)

h_conv4 = tf.nn.relu(tfFns.conv2d(h_pool3, W_conv4) + b_conv4)


# In[1]:

h_deconv1 = tf.nn.relu(tfFns.deconv2d(h_conv4,W_deconv1,output_shape=outputShape5) + b_deconv1)

h_deconv2 = tf.nn.relu(tfFns.deconv2d(h_deconv1,W_deconv2,output_shape=outputShape6) + b_deconv2)

yPred = tf.nn.sigmoid(tfFns.deconv2d(h_deconv2,W_deconv3,output_shape=outputShape7) + b_deconv3)

mse = tf.reduce_sum(tf.square(yPred - x_image))
#train_step = tf.train.AdamOptimizer(0.01).minimize(mse)


# In[ ]:

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(1000):
        batch = mnist.train.next_batch(batchSize)
        _, h_conv1 = sess.run(h_conv1,feed_dict={x: batch[0]})

