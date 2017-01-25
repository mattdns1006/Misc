
'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function
import tensorflow.contrib.layers as layers
import tensorflow as tf
#bn = layers.batch_norm
import sys
sys.path.append("/home/msmith/misc/tfFunctions")
from batchNorm import batch_norm as bn
import numpy as np
import pdb
#np.random.seed(1006)

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

# Create some wrappers for simplicity
def conv2d(x, W, b, strides):
    with tf.name_scope("convolution"):
        # Conv2D wrapper, with bias and relu activation
        with tf.name_scope("w"):
            x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        with tf.name_scope("b"):
            x = tf.nn.bias_add(x, b)

        return tf.nn.relu(x)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    with tf.name_scope("mp"):
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                              padding='SAME')

# Create model
def conv_net(x, weights, biases, dropout,batchNorm,training):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    with tf.name_scope("layer1"):
        conv1 = conv2d(x, weights['wc1'], biases['bc1'],strides=1)
        print("Using batchNorm")
        conv1 = bn(conv1,is_training=training,name="bn1")
        conv1 = maxpool2d(conv1, k=2)

    with tf.name_scope("layer2"):
        conv2 = conv2d(conv1, weights['wc2'], biases['bc2'],strides=1)
        print("Using batchNorm")
        conv2 = bn(conv2,is_training=training,name="bn2")
        conv2 = maxpool2d(conv2, k=2)

    with tf.name_scope("dense"):
        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        pdb.set_trace()
        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])

        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        # Apply Dropout
        #fc1 = tf.nn.dropout(fc1, dropout)

        # Output, class prediction
    with tf.name_scope("out"):
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
training = tf.placeholder(tf.bool)

batchNorm = 1
if batchNorm == 1:
    path = "bn/"
else:
    path = ""
pred = conv_net(x, weights, biases, keep_prob,batchNorm=batchNorm,training=training)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
variable_summaries(accuracy)

# Merged
merged = tf.summary.merge_all()

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:

    train_writer = tf.summary.FileWriter('train/{0}'.format(path),sess.graph)
    test_writer = tf.summary.FileWriter('test/{0}'.format(path),sess.graph)
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout,training:True})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc, summary = sess.run([cost, accuracy,merged], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.,training:False})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
            train_writer.add_summary(summary,step)

            #Test
            acc, summary = sess.run([accuracy,merged], feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.0,training:False})
            test_writer.add_summary(summary,step)
            print("Testing Accuracy:", acc)
        step += 1

    print("Optimization Finished!")

    # Calculate accuracy for 256 mnist test images


