import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

def crop(img,outDim):
    return tf.random_crop(img,outDim)

def brightness(img,max_delta=30):
    return tf.image.random_brightness(img,max_delta=max_delta)

def contrast(img,lower=0.7,upper=1.3):
    return tf.image.random_contrast(img,lower=lower,upper=upper)

if __name__ == "__main__":
    def show(img):
        plt.imshow(img); plt.show()
    import pdb
    import os

    path = "augmented/examples/"
    if not os.path.exists(path):
        os.makedirs(path)
    print("EXAMPLES @ {0}".format(path))
    eg = cv2.imread("../data/whale1.jpg")
    print("Original image of shape {0}.".format(eg.shape))
    x = tf.placeholder(tf.float32)
    y = crop(x,outDim=[400,400,3])
    y = brightness(y)
    y = contrast(y)
    with tf.Session() as sess:
        for i in xrange(10):
            img = y.eval(feed_dict={x:eg})
            cv2.imwrite("{0}/{1}.jpg".format(path,i),img)


