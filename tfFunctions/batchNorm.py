import tensorflow as tf  
from tensorflow.python.training import moving_averages

def batch_norm(x,is_training,name,ndim=4,decay=0.9):
    """Batch normalization."""
    params_shape = [x.get_shape()[-1]]
    with tf.variable_scope(name):
        beta = tf.get_variable( 'beta', params_shape, tf.float32, initializer=tf.constant_initializer(0.0, tf.float32))
        gamma = tf.get_variable( 'gamma', params_shape, tf.float32, initializer=tf.constant_initializer(1.0, tf.float32))

        if is_training == True:
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
    return y

