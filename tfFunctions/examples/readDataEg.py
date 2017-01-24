import glob
import cv2
import time
from tqdm import tqdm
import ipdb
import tensorflow as tf

imgPaths = glob.glob("/home/msmith/kaggle/whale/imgs/*/w1_*") # Some biggish images

filenameQ = tf.train.string_input_producer(imgPaths)
reader = tf.WholeFileReader()
key, value = reader.read(filenameQ)

img = tf.image.decode_jpeg(value)
init_op = tf.initialize_all_variables()

start = time.time()
with tf.Session() as sess:
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in tqdm(range(500)):
        img.eval().mean()
    dt = int((time.time()-start)*1000)
    print("Time = {0}".format(dt))
    

