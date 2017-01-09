import tensorflow as tf
import numpy as np
import pdb
import layers
np.set_printoptions(linewidth=100)

if __name__ == "__main__":
    inFeats = 1
    outFeats = 1
    x = np.random.rand(6*6).reshape(6,6)
    W = tf.fill([2,2,1,1],0.25)
    B = tf.fill([1],0.0)
    X = tf.placeholder(tf.float32,shape=[6,6])
    XRe = tf.reshape(X,[1,6,6,1])
    rate = 2
    Y = tf.squeeze(tf.nn.atrous_conv2d(XRe,W,rate,"VALID"))
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        yPred =Y.eval(feed_dict={X:x})
        print("in == >")
        print(x.squeeze())
        print("out == >")
        print(yPred)
        print("First ele == >")
        print(x[1,1])
        print("Equal to mean of x[:5:2,:5:2]")
        print(x[:4:2,:4:2].mean())
        pdb.set_trace()
        print(W.eval())





