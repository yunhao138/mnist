import tensorflow as tf

# 定义参数
CONV1_SIZE = 3
CONV1_DEEP = 64
CONV2_SIZE = 3
CONV2_DEEP = 128
POOL1_SIZE = 2
POOL2_SIZE = 2
FC1_SIZE = 512
INPUT_NODE = 28 * 28
OUTPUT_NODE = 10


def lenet(input_tensor=None, regularizer=None, train=True):
    """
    2个卷积层2个池化层2个全连接层
    :param input_tensor:输入图像
    :return:前向传播结果
    """
    # 通过tf.variable_scope/name_scope实现将每层分开定义，
    # 卷积层conv1，卷积核尺寸3*3，个数64
    with tf.variable_scope('layer1_conv1'):
        # 定义第一层卷积核权重矩阵
        conv1_weights = tf.get_variable('weights',
                                        shape=[CONV1_SIZE, CONV1_SIZE, 1, CONV1_DEEP],
                                        initializer=tf.contrib.layers.xavier_initializer())
        # conv1_biases = tf.get_variable('biases',
        #                          shape=[CONV1_DEEP],
        #                          initializer=tf.constant_initializer(0.0))
        # 卷积操作
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        # bn+激活函数
        relu1 = tf.nn.relu(tf.layers.batch_normalization(conv1, training=train))

    #     池化层pool1，核大小2*2，步长为2
    with tf.name_scope('layer2_pool1'):
        pool1 = tf.nn.max_pool(relu1,
                               ksize=[1, POOL1_SIZE, POOL1_SIZE, 1],
                               strides=[1, POOL1_SIZE, POOL1_SIZE, 1],
                               padding='SAME')

    # 卷积层conv2，卷积核尺寸3*3，个数128
    with tf.variable_scope('layer3_conv2'):
        conv2_weights = tf.get_variable('weights',
                                        shape=[CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                        initializer=tf.contrib.layers.xavier_initializer())

        conv2 = tf.nn.conv2d(pool1,
                             conv2_weights,
                             strides=[1, 1, 1, 1],
                             padding='SAME')
        relu2 = tf.nn.relu(tf.layers.batch_normalization(conv2, training=train))

    # 池化层pool2，核大小2*2，步长为2
    with tf.name_scope('layer4_pool2'):
        pool2 = tf.nn.max_pool(relu2,
                               ksize=[1, POOL2_SIZE, POOL2_SIZE, 1], strides=[1, POOL2_SIZE, POOL2_SIZE, 1],
                               padding='SAME')
    # 将经过卷积层提取的特征图展开成向量输入到全连接层
    pool2_shape = pool2.get_shape().as_list()
    num = pool2_shape[1] * pool2_shape[2] * pool2_shape[3]
    pool2_reshape = tf.reshape(pool2, [pool2_shape[0], num])

    # 全连接层fc1，神经元个数512
    with tf.variable_scope('layer5_fc1'):
        fc1_weights = tf.get_variable('weights', shape=[num, FC1_SIZE],
                                      initializer=tf.contrib.layers.xavier_initializer())
        # l2正则
        if regularizer is not None:
            tf.add_to_collection('loss', regularizer(fc1_weights))
        fc1_biases = tf.get_variable('biases', shape=[FC1_SIZE], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(pool2_reshape, fc1_weights) + fc1_biases)
        # if train:
        #     fc1=tf.nn.dropout(fc1,keep_prob=0.5)

    #         全连接层fc2，神经元个数10
    with tf.variable_scope('layer6_fc2'):
        fc2_weights = tf.get_variable('weights', shape=[FC1_SIZE, OUTPUT_NODE],
                                      initializer=tf.contrib.layers.xavier_initializer())
        if regularizer is not None:
            tf.add_to_collection('loss', regularizer(fc2_weights))
        fc2_biases = tf.get_variable('biases', shape=[OUTPUT_NODE], initializer=tf.constant_initializer(0.1))
        fc2 = tf.matmul(fc1, fc2_weights) + fc2_biases

    return fc2
