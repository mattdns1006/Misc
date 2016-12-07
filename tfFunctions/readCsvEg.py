import tensorflow as tf
import ipdb
import pandas as pd
import os
import matplotlib.pyplot as plt

def show(X,Y="none"):
    bs, h, w, c = X.shape 
    X = X.reshape(bs*h,w,c)
    plt.imshow(X)
    plt.title(Y)
    plt.show()


if __name__ == "__main__":

    os.chdir("/home/msmith/kaggle/whale/identifier")

    # Decode csv
    csvPath = "/home/msmith/kaggle/whale/trainCV.csv"
    df = pd.read_csv(csvPath)

    #csvPath = "/home/msmith/kaggle/whale/identifier/trainCV10.csv"
    print(df.head())
    print(df.shape)

    csvQ = tf.train.string_input_producer([csvPath])
    reader = tf.TextLineReader(skip_header_lines=1)
    k, v = reader.read(csvQ)

    defaults = [tf.constant([], shape = [1], dtype = tf.float32),
                tf.constant([], dtype = tf.string)]
                
    label, path = tf.decode_csv(v,record_defaults=defaults)

    label = tf.reshape(label,[1])

    # Define subgraph to take filename, read filename, decode and enqueue
    image_bytes = tf.read_file(path)
    decoded_img = tf.image.decode_jpeg(image_bytes)
    imageQ = tf.FIFOQueue(128,[tf.uint8,tf.float32,tf.string])
    #imageQ = tf.FIFOQueue(128,[tf.uint8,tf.float32], shapes = [[600,800,3],[1]])
    enQ_op = imageQ.enqueue([decoded_img,label,path])

    NUM_THREADS = 16
    Q = tf.train.QueueRunner(
            imageQ,
            [enQ_op]*NUM_THREADS,
            imageQ.close(),
            imageQ.close(cancel_pending_enqueues=True)
            )

    tf.train.add_queue_runner(Q)
    bS = 4
    #x,y = imageQ.dequeue_many(bS)
    x,y,path = imageQ.dequeue()


    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        count = 0
        ipdb.set_trace()
        for i in range(3000):
            x_, y_, path_ = sess.run([x,y,path])
            if x_.shape != (600,800,3):
                print(x_.shape,path_)
            count += x_.shape[0]
            if i % 100 ==0:
                print(count)
            #show(x_,y_)


