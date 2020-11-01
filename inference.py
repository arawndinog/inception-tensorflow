import tensorflow as tf 
import numpy as np
from utils import process_dataset, process_image

def load_model():
    ckpt_path = "outputs/ckpt_adam2_final/cnn_besttest"
    meta_path = ckpt_path + ".meta"

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    saver = tf.train.import_meta_graph(meta_path)
    saver.restore(sess, ckpt_path)
    x = tf.get_default_graph().get_tensor_by_name("input:0")
    output = tf.get_default_graph().get_tensor_by_name("fc/layer1/BiasAdd:0")
    # for node in sess.graph_def.node:
    #     print(node.name)
    return sess, x, output

def inference_test_set(sess, x, output):
    batch_size = 100
    is_training = tf.get_default_graph().get_tensor_by_name("is_training:0")
    y = tf.placeholder(tf.int32, [None])
    prediction = tf.to_int32(tf.argmax(output,1))
    correct_prediction = tf.equal(prediction,y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    test_data, test_label = process_dataset.extract_hdf5("outputs/casia1000_test.hdf5", False)
    test_i = 0
    test_acc_list = []
    while test_i < len(test_data):
        test_data_batch = test_data[test_i: test_i + batch_size]
        test_data_batch = process_image.reshape_img_batch_to_size(test_data_batch, (224,224))
        test_label_batch = test_label[test_i: test_i + batch_size]
        test_batch_acc = sess.run(accuracy, feed_dict = {x:test_data_batch, y:test_label_batch, is_training:False})
        test_i += batch_size
        test_acc_list.append(test_batch_acc)
        print("Done", test_i, "/", len(test_data))
    test_acc = sum(test_acc_list)/len(test_acc_list)
    return test_acc

def inference_char_img():
    import cv2
    is_training = tf.get_default_graph().get_tensor_by_name("is_training:0")
    y = tf.placeholder(tf.int32, [None])
    prediction = tf.to_int32(tf.argmax(output,1))

    img_path = None
    test_data = cv2.imread(img_path, 0)
    # preprocessing goes here
    test_data = np.expand_dims(test_data, 2)
    test_data_batch = np.expand_dims(test_data, 0)

    test_pred = sess.run(prediction, feed_dict = {x:test_data_batch, is_training:False})
    return test_pred


def main():
    sess, x, output = load_model()
    test_acc = inference_test_set(sess, x, output)
    print(test_acc)

main()