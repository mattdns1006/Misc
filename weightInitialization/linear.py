from __future__ import print_function
import tensorflow as tf
import matplotlib.pyplot as plt
import sys, pdb
sys.path.append("/home/msmith/misc/tfFunctions")
from batchNorm2 import bn
import paramCount 
import numpy as np

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

def W(nIn,nOut,weightInit,stddev=None):
    assert weightInit in ["custom","zeros","xavier","lecun_uniform"], "{0} not a valid weight intializer.".format(weightInit)
    print("Doing {0} weight initialization.".format(weightInit))
    if weightInit == "xavier":
        stddev = np.sqrt(6.0/(nIn+nOut))
        init = tf.random_normal([nIn,nOut],mean=0,stddev=stddev)
    elif weightInit == "lecun_uniform":
        scale = np.sqrt(3.0/nIn)
        init = tf.random_uniform([nIn,nOut],minval=-scale,maxval=scale)
    elif weightInit == "zeros":
        init = tf.zeros([nIn,nOut])
    elif weightInit == "custom":
        print("Custom stddev = {0:.4f}".format(stddev))
        init = tf.random_normal([nIn,nOut],mean=0,stddev=stddev)
    return tf.Variable(init)

def B(nOut):
    init = tf.constant(0.0,shape=[nOut])
    return tf.Variable(init)

def linear(X,nIn,nOut,weightInit,stddev):
    weights = W(nIn,nOut,weightInit,stddev)
    variable_summaries(weights)
    bias = B(nOut)
    variable_summaries(bias)
    return tf.matmul(X,weights) + bias

def model(X,is_training,weightInit,stddev):
    with tf.variable_scope("layer_0"):
        X = linear(X,1,10,weightInit,stddev)
        X = af(X)
    with tf.variable_scope("layer_1"):
        X = linear(X,10,10,weightInit,stddev)
        X = af(X)
    with tf.variable_scope("layer_2"):
        X = linear(X,10,10,weightInit,stddev)
        X = af(X)
    with tf.variable_scope("layer_3"):
        X = linear(X,10,10,weightInit,stddev)
        X = af(X)
    with tf.variable_scope("layer_4"):
        X = linear(X,10,1,weightInit,stddev)
    return X

if __name__ == "__main__":
    af = tf.nn.tanh
    import os
    def main():
        # Parameters
        for std in np.logspace(-2,0.6,6):
            tf.reset_default_graph()
            learning_rate = 0.001
            nEpochs = 10
            N = 500
            trainX = np.linspace(-5,5,N)
            noise = np.random.uniform(-0.15,0.15,N)
            f = lambda x: np.sin(x) + np.exp(-30*x**2) 
            y_true = f(trainX)
            trainY = y_true + noise
            weightInit = "custom"
            customStd = std 
            X = tf.placeholder(tf.float32, [None, 1])
            Y = tf.placeholder(tf.float32, [None, 1])
            is_training = tf.placeholder(tf.bool)
            keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
            YPred = model(X,True,weightInit=weightInit,stddev=customStd)
            with tf.variable_scope("loss"):
                loss = tf.reduce_mean(tf.square(tf.sub(Y,YPred)))
                variable_summaries(loss)
            trainer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
            merged = tf.summary.merge_all()
            path = "weightInitialization/linear/"
            if not os.path.exists(path):
                os.mkdir(path)
                print("Made {0}.".format(path))
            N = trainX.shape[0]
            rIdx = np.random.permutation(N)
            trainX = trainX[rIdx].reshape(N,1)
            trainY = trainY[rIdx].reshape(N,1)
            count = 0
            if weightInit == "custom":
                wp = path+weightInit+str(customStd)
            else:
                wp = path+weightInit
            x = trainX.reshape(N,1)
            y = trainY.reshape(N,1)
            with tf.Session() as sess:
                writer = tf.summary.FileWriter(wp)
                tf.global_variables_initializer().run()
                for epoch in range(nEpochs):
                    for i in xrange(0,N,5):
                        count += 1
                        x = trainX[i:i+5,:]
                        y = trainY[i:i+5,:]
                        _, summary, error = sess.run([trainer,merged,loss],feed_dict={X:x,Y:y})
                        writer.add_summary(summary,count)
                    #print("Epoch {0} error = {1}.".format(epoch,epochError.mean()))
                #egX = np.linspace(-5,5,100)
                #egY = f(egX)
                #egPred = YPred.eval(feed_dict={X:egX.reshape(100,1),is_training:False})
                #plt.title(epochError.mean())
                #plt.plot(egX,egY,'bo',egX,egPred,'k')
                #plt.show()
            sess.close()
    main()
                

