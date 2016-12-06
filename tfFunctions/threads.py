import glob
from tqdm import tqdm
import tensorflow as tf
import time

imgPaths = glob.glob("/home/msmith/kaggle/whale/imgs/*/w1_*") # Some images

filenameQ = tf.train.string_input_producer(imgPaths)

# Define a subgraph that takes a filename, reads the file, decodes it, and                                                                                     
# enqueues it.                                                                                                                                                 
filename = filenameQ.dequeue()
image_bytes = tf.read_file(filename)
decoded_image = tf.image.decode_jpeg(image_bytes)
image_queue = tf.FIFOQueue(128, [tf.uint8], None)
enqueue_op = image_queue.enqueue(decoded_image)

# Create a queue runner that will enqueue decoded images into `image_queue`.                                                                                   
NUM_THREADS = 16
queue_runner = tf.train.QueueRunner(
    image_queue,
    [enqueue_op] * NUM_THREADS,  # Each element will be run from a separate thread.                                                                                       
    image_queue.close(),
    image_queue.close(cancel_pending_enqueues=True))

# Ensure that the queue runner threads are started when we call                                                                                               
# `tf.train.start_queue_runners()` below.                                                                                                                      
tf.train.add_queue_runner(queue_runner)

# Dequeue the next image from the queue, for returning to the client.                                                                                          
img = image_queue.dequeue()

init_op = tf.initialize_all_variables()

start = time.time()
with tf.Session() as sess:
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in tqdm(range(500)):
        img.eval().mean()
    dt = int((time.time()-start)*1000)
    print("Time = {0}".format(dt))
