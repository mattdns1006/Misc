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
    csvPath = "/home/msmith/kaggle/whale/identifier/loadDataTest.csv"
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
    #imageQ = tf.FIFOQueue(128,[tf.uint8,tf.float32,tf.string])
    imageQ = tf.FIFOQueue(128,[tf.uint8,tf.float32], shapes = [[500,500,3],[1]])
    enQ_op = imageQ.enqueue([decoded_img,label,path])

    NUM_THREADS = 16
    Q = tf.train.QueueRunner(
            imageQ,
            [enQ_op]*NUM_THREADS,
            imageQ.close(),
            imageQ.close(cancel_pending_enqueues=True)
            )

    tf.train.add_queue_runner(Q)
    bS = 2
    x,y = imageQ.dequeue_many(bS)
    print(df)
    import time

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        count = 0
        try:
            while not coord.should_stop():

                if coord.should_stop():
                    break
                x_, y_ = sess.run([x,y])
                count += x_.shape[0]
                print(count)
        except tf.errors.OutOfRangeError, e:
            print("Here")
            coord.request_stop(e)
        finally:
            print("Here2")
            coord.request_stop()
            coord.join(threads)
            #show(x_,y_)


