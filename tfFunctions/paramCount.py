import tensorflow as tf

def paramCount():
    total = 0 
    for var in tf.trainable_variables():
        shape = var.get_shape()
        varParams = 1
        for dim in shape:
            varParams *= dim.value
        total += varParams
    print("Total number of trainable parameters in session ==> {0}.".format(total))

if __name__ == "__main__":
    import pdb
    w = tf.Variable(tf.random_normal([3,3,16,32]))
    w1 = tf.Variable(tf.random_normal([3,3,32,48]))
    b = tf.Variable(tf.constant(1.0,shape=[32]))
    paramCount()

