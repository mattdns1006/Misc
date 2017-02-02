from __future__ import print_function
import tensorflow.contrib.layers as layers
import tensorflow as tf
import sys
sys.path.append("/home/msmith/misc/tfFunctions")
import numpy as np
import pdb
np.random.seed(1006)

def W(shape,weightInit,scale=0.05):
    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    elif len(shape) == 4:
        fw, fh, nIn, nOut = shape
        fan_in = fw*fh*nIn # Number of weights for one output neuron in nOut
        fan_out = nOut #  
    else:
        raise ValueError, "Not a valid shape"

    if weightInit == "uniform":
        init = tf.random_uniform(shape,minval=-scale,maxval=scale)

    elif weightInit == "normal":
        init = tf.random_normal(shape,mean=0,stddev=scale)

    elif weightInit == "lecun_uniform":
        scale = np.sqrt(3.0/(fan_in))
        init = tf.random_uniform(shape,minval=-scale,maxval=scale)

    elif weightInit == "glorot_normal":
        scale = np.sqrt(2.0/(fan_in+fan_out))
        init = tf.random_normal(shape,mean=0,stddev=scale)

    elif weightInit == "glorot_uniform":
        scale = np.sqrt(6.0/(fan_in+fan_out))
        init = tf.random_uniform(shape,minval=-scale,maxval=scale)

    elif weightInit == "zeros":
        init = tf.zeros(shape)
    else:
        raise ValueError, "{0} not a valid weight intializer.".format(weightInit)

    return tf.Variable(init)


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)


def conv2d(x, W, b, strides):
    with tf.name_scope("convolution"):
        with tf.name_scope("w"):
            x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        with tf.name_scope("b"):
            x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

def maxpool2d(x, k=2):
    with tf.name_scope("mp"):
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                              padding='SAME')

def conv_net(x, weights, biases, dropout):
    max_outputs = 6 
    # Reshape input picture
    with tf.name_scope("input"):
        x = tf.reshape(x, shape=[-1, 28, 28, 1])
        tf.summary.image("x",x,max_outputs=max_outputs)

    # Convolution Layer
    with tf.name_scope("layer1"):
        conv1 = conv2d(x, weights['wc1'], biases['bc1'],strides=1)
        conv1 = maxpool2d(conv1, k=2)

        with tf.name_scope("f1"):
            for i in xrange(10):
                f = tf.split(3,16,conv1)[i]
                #w1 = tf.split(3,32,weights["wc1"])[0]
                tf.summary.image("f1_{0}".format(i),f,max_outputs=max_outputs)
                #tf.image_summary("w1",w1)


    with tf.name_scope("layer2"):
        conv2 = conv2d(conv1, weights['wc2'], biases['bc2'],strides=1)
        conv2 = maxpool2d(conv2, k=2)

        with tf.name_scope("f2"):
            for i in xrange(10):
                f = tf.split(3,16,conv2)[i]
                tf.summary.image("f2_{0}".format(i),f,max_outputs=max_outputs)

    with tf.name_scope("layer3"):
        conv3 = conv2d(conv2, weights['wc3'], biases['bc3'],strides=1)
        conv3 = maxpool2d(conv3, k=2)

        with tf.name_scope("f3"):
            for i in xrange(10):
                f = tf.split(3,16,conv3)[i]
                tf.summary.image("f3_{0}".format(i),f,max_outputs=max_outputs)

    with tf.name_scope("layer4"):
        conv4 = conv2d(conv3, weights['wc4'], biases['bc4'],strides=1)
        conv4 = maxpool2d(conv4, k=2)

        with tf.name_scope("f4"):
            for i in xrange(10):
                f = tf.split(3,16,conv4)[i]
                tf.summary.image("f4_{0}".format(i),f,max_outputs=max_outputs)


    with tf.name_scope("dense"):
        fc1 = tf.reshape(conv4, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        #fc1 = tf.nn.dropout(fc1, dropout)

        # Output, class prediction
    with tf.name_scope("out"):
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
        outVis = tf.reshape(out,[-1,10,1,1])
        tf.summary.image("outVis_{0}".format(i),outVis,max_outputs=max_outputs)
    return out


if __name__ == "__main__":
    # Import MNIST data
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    # Parameters
    learning_rate = 0.001
    training_iters = 60000
    batch_size = 128
    display_step = 10

    # Network Parameters
    n_input = 784 # MNIST data input (img shape: 28*28)
    n_classes = 10 # MNIST total classes (0-9 digits)
    dropout = 0.75 # Dropout, probability to keep units

    # tf Graph input

    # Construct model
    for weightInit in ["uniform","normal","lecun_uniform","glorot_uniform","glorot_normal","zeros"][:1]:
        tf.reset_default_graph()
        weights = {
            'wc1': W([5, 5, 1, 16],weightInit),
            'wc2': W([5, 5, 16, 16],weightInit),
            'wc3': W([5, 5, 16, 16],weightInit),
            'wc4': W([3, 3, 16, 16],weightInit),
            'wd1': W([2*2*16, 32],weightInit),
            'out': W([32, n_classes],weightInit)
        }

        biases = {
            'bc1': tf.Variable(tf.constant(0.0,shape=[16])),
            'bc2': tf.Variable(tf.constant(0.0,shape=[16])),
            'bc3': tf.Variable(tf.constant(0.0,shape=[16])),
            'bc4': tf.Variable(tf.constant(0.0,shape=[16])),
            'bd1': tf.Variable(tf.constant(0.0,shape=[32])),
            'out': tf.Variable(tf.constant(0.0,shape=[n_classes]))
        }
        x = tf.placeholder(tf.float32, [None, n_input])
        y = tf.placeholder(tf.float32, [None, n_classes])
        keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
        training = tf.placeholder(tf.bool)

        pred = conv_net(x, weights, biases, keep_prob)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        # Evaluate model
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        variable_summaries(accuracy)

        merged = tf.summary.merge_all()
        init = tf.initialize_all_variables()
        path = "conv/"

        with tf.Session() as sess:

            #train_writer = tf.summary.FileWriter('{0}/{1}/train'.format(path,weightInit),sess.graph)
            test_writer = tf.summary.FileWriter('{0}/{1}/test'.format(path,weightInit),sess.graph)
            sess.run(init)
            step = 1
            # Keep training until reach max iterations
            while step * batch_size < training_iters:
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                # Run optimization op (backprop)
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout,training:True})
                if step % display_step == 0:
                    # Calculate batch loss and accuracy
                    #loss, acc, summary = sess.run([cost, accuracy,merged], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.,training:False})
                    #print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
                    #train_writer.add_summary(summary,step)

                    #Test
                    acc, summary = sess.run([accuracy,merged], feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.0,training:False})
                    test_writer.add_summary(summary,step)
                    print("Testing Accuracy:", acc)
                step += 1

            print("Optimization Finished!")

        # Calculate accuracy for 256 mnist test images


