from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
import tensorflow as tf

def bn(x,is_training,name):
    bn_train = batch_norm(x, decay=0.9, center=True, scale=True,
    updates_collections=None,
    is_training=True,
    reuse=None, # is this right?
    trainable=True,
    scope=name)
    bn_inference = batch_norm(x, decay=0.9, center=True, scale=True,
    updates_collections=None,
    is_training=False,
    reuse=True, # is this right?
    trainable=True,
    scope=name)
    z = tf.cond(is_training, lambda: bn_train, lambda: bn_inference)
    return z

if __name__ == "__main__":
    print("Example")

    import numpy as np
    import scipy.stats as stats
    np.set_printoptions(linewidth=200,precision=2)
    np.random.seed(1006)
    import pdb

    nFeats = 5
    X = tf.placeholder(tf.float32,[None,nFeats])
    is_training = tf.placeholder(tf.bool,name="is_training")
    Y = bn(X,is_training=is_training,name="bn")
    mvn = stats.multivariate_normal([0,10,20,30,40])
    bs = 4
    def bn_(batch):
        return (x - x.mean(0))/(x.std(0) + 0.001)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in xrange(10):
            x = mvn.rvs(bs)
            y = Y.eval(feed_dict={X:x, is_training.name: True})

        for i in xrange(10):
            x = mvn.rvs(1).reshape(1,-1)
            y = Y.eval(feed_dict={X:x, is_training.name: False})
            print(y)
