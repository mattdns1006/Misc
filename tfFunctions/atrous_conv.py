import tensorflow as tf
import numpy as np

def weightVar(shape,const):
    initial = tf.constant(const,shape=shape)
    return tf.Variable(initial)

def biasVar(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

def atrous_conv(x,weights,rate,padding):
    return tf.nn.atrous_conv2d(value=x,filters=weights,rate=rate,padding=padding)

if __name__ == "__main__":
    np.set_printoptions(linewidth=200,precision=3)
    import pdb
    ks = 3
    rate = 2
    W = weightVar([ks,ks,1,1],1.0/(ks**2))
    B = biasVar([1])
    w = 16 
    X = tf.placeholder(tf.float32,[w,w])
    Xre = tf.reshape(X,[1,w,w,1])
    Y = atrous_conv(Xre,W,rate,"VALID")
    Y = tf.squeeze(Y)

    def check(x,ks = ks, rate = rate):
        effectiveHW = (ks + (ks-1)*(rate-1))
        print(effectiveHW)
        return x[:effectiveHW,:effectiveHW]

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in xrange(10):
            x = np.arange(w**2).reshape(w,w)
            y = Y.eval(feed_dict={X:x})
            print(x)
            print(y)
            pdb.set_trace()
