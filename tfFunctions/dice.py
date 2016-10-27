import tensorflow as tf

def dice(yPred,yTruth,thresh):
    smooth = tf.constant(1.0)
    threshold = tf.constant(thresh)
    yPredThresh = tf.to_float(tf.greater_equal(yPred,threshold))
    mul = tf.mul(yPredThresh,yTruth)
    intersection = 2*tf.reduce_sum(mul) + smooth
    union = tf.reduce_sum(yPredThresh) + tf.reduce_sum(yTruth) + smooth
    dice = intersection/union
    return dice, yPredThresh

if __name__ == "__main__":
    import ipdb
    with tf.Session() as sess:

        thresh = 0.5
        print("Dice example")
        yPred = tf.constant([0.1,0.9,0.7,0.3,0.1,0.1,0.9,0.9,0.1],shape=[3,3])
        yTruth = tf.constant([0.0,1.0,1.0,0.0,0.0,0.0,1.0,1.0,1.0],shape=[3,3])
        diceScore, yPredThresh= dice(yPred=yPred,yTruth=yTruth,thresh= thresh)

        diceScore_ , yPredThresh_ , yPred_, yTruth_ = sess.run([diceScore,yPredThresh,yPred, yTruth])


        print("\nTruth = ")
        print(yTruth_)

        print("\nPrediction = ")
        print(yPred_)


        print("\nPrediction thresholded at {0} = \n".format(thresh))
        print(yPredThresh_)

        print("\nScore = {0}".format(diceScore_))
        ipdb.set_trace()
