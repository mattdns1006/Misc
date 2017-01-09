import tensorflow as tf
import numpy as np
import ipdb
from tensorflow.contrib.layers import layers

if __name__ == "__main__":
    bn = layers.batch_norm
    nFeats = 3
    nObs = 100
    xTr = np.random.rand(nObs,nFeats)
    xTe = np.random.rand(1,nFeats)
    bnTrain = tf.placeholder(tf.bool)
    X = tf.placeholder(tf.float32,[None,nFeats])
    Y = bn(X,nFeats,is_training=bnTrain)
    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init_op)
        yTr_ = Y.eval(feed_dict={X:xTr,bnTrain:1})
        yTe_ = Y.eval(feed_dict={X:xTe,bnTrain:0})
        ipdb.set_trace()
        print("here")
