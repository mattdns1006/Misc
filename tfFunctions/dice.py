import tensorflow as tf
import numpy as np

'''
Only suitable for 4-dimensional tensors (batch mode) at the moment.
'''

def dice(yPred,yTruth,threshold=0.5):
    smooth = tf.constant(1.0)
    threshold = tf.constant(threshold)
    yPredThresh = tf.to_float(tf.greater_equal(yPred,threshold))
    mul = tf.mul(yPredThresh,yTruth)
    intersection = 2*tf.reduce_sum(mul, [1,2,3]) + smooth
    union = tf.reduce_sum(yPredThresh, [1,2,3]) + tf.reduce_sum(yTruth, [1,2,3]) + smooth
    dice = intersection/union
    return dice

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
    import cv2, glob, sys
    import matplotlib.pyplot as plt
    sys.path.append("../py/")
    from hStackBatch import hStackBatch
    def show(img):
        img = hStackBatch(img)
        plt.imshow(img)
        plt.show()

    bS = 5
    Y_ = np.empty((bS,19,29,3))
    YPred_ = Y_.copy()
    sd = 1

    paths = glob.glob("/home/msmith/kaggle/whale/imgs/whale_11099/m1_*.jpg")
    for i in range(bS):
        yPred_ = cv2.imread(paths[i])/255.0

        Y_[i] = cv2.imread(paths[i])/255.0

        r, c, col = yPred_.shape
        rx = np.random.normal(0,sd)
        ry = np.random.normal(0,sd)
        M = np.float32([[1,0,rx],[0,1,ry]])
        yPred_ = cv2.warpAffine(yPred_,M,(c,r))
        YPred_[i] = yPred_


    Y_[Y_>0.2] = 1.0
    Y_[Y_<=0.2] = 0.0

    with tf.Session() as sess:

        nThreshs = 20
        thresh = 0.5
        print("Dice example")
        yPred = tf.placeholder(tf.float32,shape=[None,None,None,None])
        yTruth = tf.placeholder(tf.float32,shape=[None,None,None,None])

        threshs = np.linspace(0.1,0.9,nThreshs)
        diceScore, _= dice(yPred=yPred,yTruth=yTruth,threshold=thresh)
        diceScoreROC = diceROC(yPred=yPred,yTruth=yTruth,thresholds=threshs)

        diceScoreROC_ , = sess.run([diceScoreROC],feed_dict={yPred:YPred_,yTruth:Y_})
        print("\nScore = {0}".format(diceScoreROC_.mean(0)))
        show(np.hstack((Y_,YPred_)))





