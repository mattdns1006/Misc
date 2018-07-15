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

def avgp(x,kSize,stride):
    return tf.nn.avg_pool(x,[1,kSize,kSize,1],strides=[1,stride,stride,1],padding = "VALID")

if __name__ == "__main__":
    import numpy as np
    np.set_printoptions(linewidth=100,precision=3)
    import pdb
    X = tf.placeholder(tf.float32,[1,7,7,1])
    Y = avgp(X,7,1)
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for i in xrange(10):
            x = np.random.random((1,7,7,1))
            y = Y.eval(feed_dict={X:x})
            x = x.squeeze()
            y = y.squeeze()
            print(x)
            print("\n")
            print(y)
            pdb.set_trace()

