import tensorflow as tf

def weightVar(shape,stddev=0.01):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)

def biasVar(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def mp(x,kS,stride):
    return tf.nn.max_pool(x, ksize=[1, kS, kS, 1],strides=[1, stride, stride, 1], padding='SAME')

def deconv2d(x,W,outputShape,stride):
    return tf.nn.conv2d_transpose(x, W, output_shape=outputShape, strides=[1, stride, stride, 1], padding='SAME')

def linear(x,W,b):
    return tf.matmul(x,W) + b

def bn(shape, input):
    eps = 1e-5
    weight_variable = lambda shape: tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    gamma = weight_variable([shape])
    beta = weight_variable([shape])
    mean, variance = tf.nn.moments(input, [0])
    return gamma * (input - mean) / tf.sqrt(variance + eps) + beta
