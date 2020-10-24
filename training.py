import tensorflow as tf 
import numpy as np
import cv2
import inception
from utils import process_dataset, process_image
from multiprocessing import Queue, Process

def main():
    train_data, train_label = process_dataset.extract_hdf5("/home/adrianwong/Projects/ML_localdata/Dataset/HDF5/CASIA_SC_L_A_label_00000_03880.hdf5")
    train_indices_partial = train_label < 1000
    train_data = train_data[train_indices_partial]
    train_label = train_label[train_indices_partial]
    batch_size = 100

    test_data, test_label = process_dataset.extract_hdf5("/home/adrianwong/Projects/ML_localdata/Dataset/HDF5/CASIA_SC_C_0.hdf5")
    test_indices_partial = test_label < 1000
    test_data = test_data[test_indices_partial]
    test_label = test_label[test_indices_partial]
    test_data = process_image.reshape_img_batch_to_size(test_data, (224,224))

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        x = tf.placeholder(tf.float32, [None, 224, 224, 3], name="input")
        y = tf.placeholder(tf.int32, [None])
        learning_rate = tf.placeholder(tf.float32, name="learning_rate")
        is_training = tf.placeholder(tf.bool, name="is_training")
        dense1, dense4a, dense4d = inception.inception1(x, is_training)

        cost_dense1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=dense1,labels=y))
        cost_dense4a = 0.3 * tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=dense4a,labels=y))
        cost_dense4d = 0.3 * tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=dense4d,labels=y))
        cost = cost_dense1 + cost_dense4a + cost_dense4d

        # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(cost)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        prediction = tf.to_int32(tf.argmax(dense1,1))
        correct_prediction = tf.equal(prediction,y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


        train_data_q = Queue(1000)
        enque_train_data_process1 = Process(target=enque_batch_train_data_process, args=(train_data_q, train_data, train_label, batch_size))
        enque_train_data_process1.start()

        sess.run(tf.global_variables_initializer())
        # print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
        saver = tf.train.Saver(max_to_keep=None)
        saver.restore(sess, "outputs/cnn_latest")
        saver_iter = 5000

        train_acc = 0
        train_loss = 100000
        test_acc = 0

        i = 0
        while True:
            train_data_batch, train_label_batch = train_data_q.get()
            _, train_loss = sess.run([optimizer, cost], feed_dict = {x:train_data_batch, y:train_label_batch, learning_rate: 0.0001, is_training:True})
            print("Iteration:", i)
            print("     Training loss:", train_loss, "Last training acc:", train_acc, "Last test acc:", test_acc)

            if (i % saver_iter == 0) and (i != 0):
                train_acc = sess.run(accuracy, feed_dict = {x:train_data_batch, y:train_label_batch, is_training:False})

                test_i = 0
                test_acc_list = []
                while test_i < len(test_data):
                    test_data_batch = test_data[test_i: test_i + batch_size]
                    test_label_batch = test_label[test_i: test_i + batch_size]
                    test_batch_acc = sess.run(accuracy, feed_dict = {x:test_data_batch, y:test_label_batch, is_training:False})
                    test_i += batch_size
                    test_acc_list.append(test_batch_acc)
                test_acc = sum(test_acc_list)/len(test_acc_list)

                saver.save(sess, "outputs/cnn_latest", global_step=None)
                print(train_loss, train_acc, test_acc)
                print("checkpoint saved")

            i += 1


def enque_batch_train_data_process(q, train_data, train_label, batch_size):
    i = 0
    train_data_indices = np.arange(len(train_data))
    np.random.shuffle(train_data_indices)
    while True:
        if i < len(train_data):
            batch_indices = train_data_indices[i:i+batch_size]
            train_data_batch = train_data[batch_indices]
            train_label_batch = train_label[batch_indices]

            train_data_batch = process_image.reshape_img_batch_to_size(train_data_batch, (224,224))

            q.put((train_data_batch,train_label_batch))
            i += batch_size
        else:
            i = 0
            np.random.shuffle(train_data_indices)
            print("data shuffled")
main()