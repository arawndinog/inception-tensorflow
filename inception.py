import tensorflow as tf
# last update 20201023

def inception1_block(x: tf.Tensor, 
                    input_filters=192,
                    conv1_filters=64, 
                    conv3_reduce_filters=96, conv3_filters=128, 
                    conv5_reduce_filters=16, conv5_filters=32, 
                    pool_proj_filters=32
                    ):
    with tf.variable_scope('conv1x1_block'):
        W_conv1 = tf.Variable(tf.truncated_normal([1, 1, input_filters, conv1_filters], stddev=0.1), name="weights")
        b_conv1 = tf.Variable(tf.constant(value=0.1, shape=[conv1_filters]), name='bias')
        conv1 = tf.nn.bias_add(tf.nn.conv2d(x, W_conv1, strides=[1, 1, 1, 1], padding='SAME'), b_conv1)
        conv1_active = tf.nn.relu(conv1)
    with tf.variable_scope('conv3x3_block'):
        # 3x3 conv reduce
        W_conv3_reduce = tf.Variable(tf.truncated_normal([1, 1, input_filters, conv3_reduce_filters], stddev=0.1), name="weights")
        b_conv3_reduce = tf.Variable(tf.constant(value=0.1, shape=[conv3_reduce_filters]), name='bias')
        conv3_reduce = tf.nn.bias_add(tf.nn.conv2d(x, W_conv3_reduce, strides=[1, 1, 1, 1], padding='SAME'), b_conv3_reduce)
        conv3_reduce_active = tf.nn.relu(conv3_reduce)
        # 3x3 conv
        W_conv3 = tf.Variable(tf.truncated_normal([3, 3, conv3_reduce_filters, conv3_filters], stddev=0.1), name="weights")
        b_conv3 = tf.Variable(tf.constant(value=0.1, shape=[conv3_filters]), name='bias')
        conv3 = tf.nn.bias_add(tf.nn.conv2d(conv3_reduce_active, W_conv3, strides=[1, 1, 1, 1], padding='SAME'), b_conv3)
        conv3_active = tf.nn.relu(conv3)
    with tf.variable_scope('conv5x5_block'):
        # 5x5 conv reduce
        W_conv5_reduce = tf.Variable(tf.truncated_normal([1, 1, input_filters, conv5_reduce_filters], stddev=0.1), name="weights")
        b_conv5_reduce = tf.Variable(tf.constant(value=0.1, shape=[conv5_reduce_filters]), name='bias')
        conv5_reduce = tf.nn.bias_add(tf.nn.conv2d(x, W_conv5_reduce, strides=[1, 1, 1, 1], padding='SAME'), b_conv5_reduce)
        conv5_reduce_active = tf.nn.relu(conv5_reduce)
        # 5x5 conv
        W_conv5 = tf.Variable(tf.truncated_normal([5, 5, conv5_reduce_filters, conv5_filters], stddev=0.1), name="weights")
        b_conv5 = tf.Variable(tf.constant(value=0.1, shape=[conv5_filters]), name='bias')
        conv5 = tf.nn.bias_add(tf.nn.conv2d(conv5_reduce_active, W_conv5, strides=[1, 1, 1, 1], padding='SAME'), b_conv5)
        conv5_active = tf.nn.relu(conv5)
    with tf.variable_scope('poolproj_block'):
        pool_proj = tf.nn.max_pool(x, ksize=[1,3,3,1], strides=[1,1,1,1], padding='SAME')
        W_pool_proj_reduce = tf.Variable(tf.truncated_normal([1, 1, input_filters, pool_proj_filters], stddev=0.1), name="weights")
        b_pool_proj_reduce = tf.Variable(tf.constant(value=0.1, shape=[pool_proj_filters]), name='bias')
        pool_proj_reduce = tf.nn.bias_add(tf.nn.conv2d(pool_proj, W_pool_proj_reduce, strides=[1, 1, 1, 1], padding='SAME'), b_pool_proj_reduce)
        pool_proj_reduce_active = tf.nn.relu(pool_proj_reduce)

    depth_concat = tf.concat([conv1_active, conv3_active, conv5_active, pool_proj_reduce_active], axis=3, name="concat")
    return depth_concat

def inception1(x: tf.Tensor, is_training: tf.bool):
    assert x.get_shape()[1:]==(224,224,3)
    with tf.variable_scope('cnn'):
        x = tf.truediv(x, 255.0)

        with tf.variable_scope('layer1'):
            W = tf.Variable(tf.truncated_normal([7, 7, 3, 64], stddev=0.1), name="weights")
            b = tf.Variable(tf.constant(value=0.1, shape=[64]), name='bias')
            conv1 = tf.nn.bias_add(tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME'), b)
            conv1_active = tf.nn.relu(conv1)
            assert conv1_active.get_shape()[1:]==(112,112,64)
            conv1_pool = tf.nn.max_pool(conv1_active, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
            assert conv1_pool.get_shape()[1:]==(56,56,64)
            conv1_norm = tf.nn.local_response_normalization(conv1_pool)

        with tf.variable_scope('layer2a'):
            W = tf.Variable(tf.truncated_normal([1, 1, 64, 64], stddev=0.1), name="weights")
            b = tf.Variable(tf.constant(value=0.1, shape=[64]), name='bias')
            conv2a = tf.nn.bias_add(tf.nn.conv2d(conv1_norm, W, strides=[1, 1, 1, 1], padding='SAME'),b)
            conv2a_active = tf.nn.relu(conv2a)

        with tf.variable_scope('layer2b'):
            W = tf.Variable(tf.truncated_normal([3, 3, 64, 192], stddev=0.1), name="weights")
            b = tf.Variable(tf.constant(value=0.1, shape=[192]), name='bias')
            conv2b = tf.nn.bias_add(tf.nn.conv2d(conv2a_active, W, strides=[1, 1, 1, 1], padding='SAME'),b)
            conv2b_active = tf.nn.relu(conv2b)
            assert conv2b_active.get_shape()[1:]==(56,56,192)
            conv2b_norm = tf.nn.local_response_normalization(conv2b_active)
            conv2b_pool = tf.nn.max_pool(conv2b_norm, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
            assert conv2b_pool.get_shape()[1:]==(28,28,192)

        with tf.variable_scope('layer3a'):
            conv3a = inception1_block(
                    x=conv2b_pool, input_filters=192, 
                    conv1_filters=64, 
                    conv3_reduce_filters=96, conv3_filters=128, 
                    conv5_reduce_filters=16, conv5_filters=32, 
                    pool_proj_filters=32)
            assert conv3a.get_shape()[1:]==(28,28,256)

        with tf.variable_scope('layer3b'):
            conv3b = inception1_block(
                    x=conv3a, input_filters=256, 
                    conv1_filters=128, 
                    conv3_reduce_filters=128, conv3_filters=192, 
                    conv5_reduce_filters=32, conv5_filters=96, 
                    pool_proj_filters=64)
            assert conv3b.get_shape()[1:]==(28,28,480)

        conv3b_pool = tf.nn.max_pool(conv3b, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
        assert conv3b_pool.get_shape()[1:]==(14,14,480)

        with tf.variable_scope('layer4a'):
            conv4a = inception1_block(
                    x=conv3b_pool, input_filters=480, 
                    conv1_filters=192, 
                    conv3_reduce_filters=96, conv3_filters=208, 
                    conv5_reduce_filters=16, conv5_filters=48, 
                    pool_proj_filters=64)
            assert conv4a.get_shape()[1:]==(14,14,512)

        with tf.variable_scope('layer4a_aux_conv'):
            conv4a_pool = tf.nn.avg_pool(conv4a, ksize=[1,5,5,1], strides=[1,3,3,1], padding='VALID')
            assert conv4a_pool.get_shape()[1:]==(4,4,512)
            W = tf.Variable(tf.truncated_normal([1, 1, 512, 128], stddev=0.1), name="weights")
            b = tf.Variable(tf.constant(value=0.1, shape=[128]), name='bias')
            conv4a_aux = tf.nn.bias_add(tf.nn.conv2d(conv4a_pool, W, strides=[1, 1, 1, 1], padding='SAME'),b)
            conv4a_aux_active = tf.nn.relu(conv4a_aux)
            assert conv4a_aux_active.get_shape()[1:]==(4,4,128)

        with tf.variable_scope('layer4a_aux_dense1'):
            W = tf.Variable(tf.truncated_normal([128*4*4, 1024], stddev=0.1), name="weights")
            b = tf.Variable(tf.constant(value=0.1, shape=[1024]), name='bias')
            conv4a_aux_flatten = tf.reshape(conv4a_aux_active,[-1,W.get_shape().as_list()[0]])
            assert conv4a_aux_flatten.get_shape()[1:]==(128*4*4)
            dense4a1 = tf.nn.bias_add(tf.matmul(conv4a_aux_flatten, W), b)
            dense4a1_drop = tf.layers.dropout(dense4a1, rate=0.7, training=is_training)
            assert dense4a1_drop.get_shape()[1:]==(1024)

        with tf.variable_scope('layer4a_aux_dense2'):
            W = tf.Variable(tf.truncated_normal([1024, 1000], stddev=0.1), name="weights")
            b = tf.Variable(tf.constant(value=0.1, shape=[1000]), name='bias')
            dense4a2 = tf.nn.bias_add(tf.matmul(dense4a1_drop, W), b)
            assert dense4a2.get_shape()[1:]==(1000)

        with tf.variable_scope('layer4b'):
            conv4b = inception1_block(
                    x=conv4a, input_filters=512, 
                    conv1_filters=160, 
                    conv3_reduce_filters=112, conv3_filters=224, 
                    conv5_reduce_filters=24, conv5_filters=64, 
                    pool_proj_filters=64)
            assert conv4b.get_shape()[1:]==(14,14,512)

        with tf.variable_scope('layer4c'):
            conv4c = inception1_block(
                    x=conv4b, input_filters=512, 
                    conv1_filters=128, 
                    conv3_reduce_filters=128, conv3_filters=256, 
                    conv5_reduce_filters=24, conv5_filters=64, 
                    pool_proj_filters=64)
            assert conv4c.get_shape()[1:]==(14,14,512)

        with tf.variable_scope('layer4d'):
            conv4d = inception1_block(
                    x=conv4c, input_filters=512, 
                    conv1_filters=112, 
                    conv3_reduce_filters=144, conv3_filters=288, 
                    conv5_reduce_filters=32, conv5_filters=64, 
                    pool_proj_filters=64)
            assert conv4d.get_shape()[1:]==(14,14,528)

        with tf.variable_scope('layer4d_aux_conv'):
            conv4d_pool = tf.nn.avg_pool(conv4d, ksize=[1,5,5,1], strides=[1,3,3,1], padding='VALID')
            assert conv4d_pool.get_shape()[1:]==(4,4,528)
            W = tf.Variable(tf.truncated_normal([1, 1, 528, 128], stddev=0.1), name="weights")
            b = tf.Variable(tf.constant(value=0.1, shape=[128]), name='bias')
            conv4d_aux = tf.nn.bias_add(tf.nn.conv2d(conv4d_pool, W, strides=[1, 1, 1, 1], padding='SAME'),b)
            conv4d_aux_active = tf.nn.relu(conv4d_aux)
            assert conv4d_aux_active.get_shape()[1:]==(4,4,128)

        with tf.variable_scope('layer4d_aux_dense1'):
            W = tf.Variable(tf.truncated_normal([128*4*4, 1024], stddev=0.1), name="weights")
            b = tf.Variable(tf.constant(value=0.1, shape=[1024]), name='bias')
            conv4d_aux_flatten = tf.reshape(conv4d_aux_active,[-1,W.get_shape().as_list()[0]])
            assert conv4d_aux_flatten.get_shape()[1:]==(128*4*4)
            dense4d1 = tf.nn.bias_add(tf.matmul(conv4d_aux_flatten, W), b)
            dense4d1_drop = tf.layers.dropout(dense4d1, rate=0.7, training=is_training)
            assert dense4d1_drop.get_shape()[1:]==(1024)

        with tf.variable_scope('layer4d_aux_dense2'):
            W = tf.Variable(tf.truncated_normal([1024, 1000], stddev=0.1), name="weights")
            b = tf.Variable(tf.constant(value=0.1, shape=[1000]), name='bias')
            dense4d2 = tf.nn.bias_add(tf.matmul(dense4d1_drop, W), b)
            assert dense4d2.get_shape()[1:]==(1000)

        with tf.variable_scope('layer4e'):
            conv4e = inception1_block(
                    x=conv4d, input_filters=528, 
                    conv1_filters=256, 
                    conv3_reduce_filters=160, conv3_filters=320, 
                    conv5_reduce_filters=32, conv5_filters=128, 
                    pool_proj_filters=128)
            assert conv4e.get_shape()[1:]==(14,14,832)

        conv4e_pool = tf.nn.max_pool(conv4e, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
        assert conv4e_pool.get_shape()[1:]==(7,7,832)

        with tf.variable_scope('layer5a'):
            conv5a = inception1_block(
                    x=conv4e_pool, input_filters=832, 
                    conv1_filters=256, 
                    conv3_reduce_filters=160, conv3_filters=320, 
                    conv5_reduce_filters=32, conv5_filters=128, 
                    pool_proj_filters=128)
            assert conv5a.get_shape()[1:]==(7,7,832)

        with tf.variable_scope('layer5b'):
            conv5b = inception1_block(
                    x=conv5a, input_filters=832, 
                    conv1_filters=384, 
                    conv3_reduce_filters=192, conv3_filters=384, 
                    conv5_reduce_filters=48, conv5_filters=128, 
                    pool_proj_filters=128)
            assert conv5b.get_shape()[1:]==(7,7,1024)

        conv5b_pool = tf.nn.avg_pool(conv5b, ksize=[1,7,7,1], strides=[1,1,1,1], padding='VALID')
        assert conv5b_pool.get_shape()[1:]==(1,1,1024)
        conv5b_drop = tf.layers.dropout(conv5b_pool, rate=0.4, training=is_training)

    with tf.variable_scope('fc'):
        with tf.variable_scope('layer1'):
            W = tf.Variable(tf.truncated_normal([1024, 1000], stddev=0.1), name="weights")
            b = tf.Variable(tf.constant(value=0.1, shape=[1000]), name='bias')
            conv5b_flatten = tf.reshape(conv5b_drop,[-1,W.get_shape().as_list()[0]])
            assert conv5b_flatten.get_shape()[1:]==(1024)
            dense1 = tf.nn.bias_add(tf.matmul(conv5b_flatten, W), b)
            assert dense1.get_shape()[1:]==(1000)
            # dense1_active = tf.nn.relu(dense1)

    return dense1, dense4a2, dense4d2