import tensorflow as tf
import numpy as np

'''
Only suitable for 3-dimensional tensors at the moment.
'''

def dice(yPred,yTruth,threshold=0.5):
    smooth = tf.constant(1.0)
    threshold = tf.constant(threshold)
    yPredThresh = tf.to_float(tf.greater_equal(yPred,threshold))
    mul = tf.mul(yPredThresh,yTruth)
    intersection = 2*tf.reduce_sum(mul, [0,1,2]) + smooth
    union = tf.reduce_sum(yPredThresh, [0,1,2]) + tf.reduce_sum(yTruth, [0,1,2]) + smooth
    dice = intersection/union
    return dice, yPredThresh

def diceROC(yPred,yTruth,thresholds=np.linspace(0.1,0.9,20)):
    thresholds = thresholds.astype(np.float32)
    nThreshs = thresholds.size
    nDims = len(yPred.get_shape())
    yPred = tf.expand_dims(yPred,nDims)
    yTruth = tf.expand_dims(yTruth,nDims)
    dims = np.ones(nDims+1).astype(np.uint32)
    dims[-1] = nThreshs
    yPredT = tf.tile(yPred,dims)
    yTruthT = tf.tile(yTruth,dims)
    diceScores, _ = dice(yPred=yPred, yTruth=yTruth, threshold = thresholds)
    return diceScores

if __name__ == "__main__":
    import ipdb
    with tf.Session() as sess:

        nThreshs = 20
        thresh = 0.5
        print("Dice example")
        yPred = tf.constant([0.1,0.9,0.7,0.3,0.1,0.1,0.9,0.9,0.1,
                             0.1,0.9,0.7,0.3,0.1,0.1,0.9,0.9,0.1,
                             0.1,0.9,0.7,0.3,0.1,0.1,0.9,0.9,0.1],shape=[3,3,3])
        yTruth = tf.constant([0.0,1.0,1.0,0.0,0.0,0.0,1.0,1.0,1.0,
                              0.0,1.0,1.0,0.0,0.0,0.0,1.0,1.0,1.0,
                              0.0,1.0,1.0,0.0,0.0,0.0,1.0,1.0,1.0],shape=[3,3,3])

        threshs = np.linspace(0.1,0.9,nThreshs)
        diceScore, _= dice(yPred=yPred,yTruth=yTruth,threshold=thresh)
        diceScoreROC = diceROC(yPred=yPred,yTruth=yTruth,thresholds=threshs)

        diceScore_ = diceScore.eval()
        diceScoreROC_ , yPred_, yTruth_ = sess.run([diceScoreROC, yPred, yTruth])

        ipdb.set_trace()

        print("\nScore = {0}".format(diceScoreROC_))

