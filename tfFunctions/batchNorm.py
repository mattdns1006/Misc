import tensorflow as tf  
from tensorflow.python.training import moving_averages


def batch_norm(x,is_training,name,ndim=4,decay=0.9,example=0):
    """Batch normalization."""
    params_shape = [x.get_shape()[-1]]
    with tf.variable_scope(name):
        beta = tf.get_variable( 'beta', params_shape, tf.float32, initializer=tf.constant_initializer(0.0, tf.float32))
        gamma = tf.get_variable( 'gamma', params_shape, tf.float32, initializer=tf.constant_initializer(1.0, tf.float32))

        if is_training == True:
            print("here")
            if ndim == 4:
                dim = [0,1,2]
            elif ndim == 2:
                dim = [0]
            mean, variance = tf.nn.moments(x, dim, name='moments')

            moving_mean = tf.get_variable( 'moving_mean', params_shape, tf.float32, initializer=tf.constant_initializer(0.0, tf.float32), trainable=False)
            moving_variance = tf.get_variable( 'moving_variance', params_shape, tf.float32, initializer=tf.constant_initializer(1.0, tf.float32), trainable=False)

            moving_mean = moving_averages.assign_moving_average( moving_mean, mean, decay)
            moving_variance = moving_averages.assign_moving_average( moving_variance, variance, decay)
        else:
            mean = tf.get_variable( 'moving_mean', params_shape, tf.float32, initializer=tf.constant_initializer(0.0, tf.float32), trainable=False)
            variance = tf.get_variable( 'moving_variance', params_shape, tf.float32, initializer=tf.constant_initializer(1.0, tf.float32), trainable=False)
            # elipson used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
        y = tf.nn.batch_normalization( x, mean, variance, beta, gamma, 0.001) 
        y.set_shape(x.get_shape()) 
    if example == 1:
        return y, mean, variance, moving_mean, moving_variance, beta, gamma
    else:
        return y




if __name__ == "__main__":
    import numpy as np
    import scipy.stats as stats
    np.set_printoptions(linewidth=200,precision=3)
    np.random.seed(1006)
    import pdb
    from tensorflow.contrib.layers import batch_norm
    print("Example")
    X = tf.placeholder(tf.float32,[None,5])
    #Y, mean, var, mmean, mvar,beta,gamma = batch_norm(X,is_training=True,ndim=2,name="bn",example=1)
    mvn = stats.multivariate_normal([0,10,20,30,40])
    bs = 4
    def bn(batch):
        return (x - x.mean(0))/(x.std(0) + 0.001)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in xrange(10):
            x = mvn.rvs(bs)
            y, mean_, var_, mmean_, mvar_,b, g = sess.run([Y,mean,var,mmean,mvar,beta,gamma],feed_dict={X:x})
            print("output = {0} \n, mean = {1}, \n var = {2}, \n mmean = {3}, \n \
                mvar = {4} \n Beta = {5}, Gamma = {6}.".format(y,mean_,var_,mmean_,mvar_,b,g))
            pdb.set_trace()


