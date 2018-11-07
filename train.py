import tensorflow as tf
import inference, os
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 256


def train():
    mnist = input_data.read_data_sets('mnist/', one_hot=True)
    # 输入数据和标签的占位符
    x = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 28, 28, 1])
    y_ = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 10])
    # l2正则
    l2_regularizer = tf.contrib.layers.l2_regularizer(0.0001)
    # 前向传播
    logits = inference.lenet(x, l2_regularizer)
    # 定义交叉熵loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=tf.argmax(y_, 1), logits=logits)
    # ‘loss’集合中为l2正则项，与交叉熵相加为总loss
    losses = loss + tf.add_n(tf.get_collection('loss'))
    # 用tensorboard查看损失函数变化趋势
    tf.summary.scalar('loss', losses)
    # 定义一个全局步骤，表示迭代步数，不可训练
    global_step = tf.Variable(0, trainable=False)
    # 定义optimizer
    opt = tf.train.AdamOptimizer()
    # 此api可以控制依赖,括号中的操作会保证先执行,这里用来使bn参数更新
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op = opt.minimize(losses, global_step=global_step)
    # 定义accuracy
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1)), tf.float32))
    # 用tensorboard查看精确度的变化趋势
    tf.summary.scalar('accuracy', accuracy)
    # Saver类是tf中用来保存模型的
    saver = tf.train.Saver()
    # 定义显卡相关设置
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    merged = tf.summary.merge_all()
    # 创建一个会话，tf所有的操作都是定义在图上的，要执行这些操作必须在会话中通过run命令
    with tf.Session(config=config) as sess:
        # 所有变量初始化，包括weights等，必要步骤
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter('log')
        # 50000个batches
        for i in range(50000):
            # mnist是整合到tf中的，因此tf提供了一个api用于获取下一个·batch
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            # reshape成矩阵形式
            xs = xs.reshape([BATCH_SIZE, 28, 28, 1])
            # 开始训练,并计算liss、accuracy
            _, loss, acc, step, merge = sess.run([train_op, losses, accuracy, global_step, merged],
                                                 feed_dict={x: xs, y_: ys})
            # 每隔200个batches保存一次
            if i % 200 == 0:
                # 对比有哪些bn层参数得到更新
                print(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
                print('After %s steps,loss is %s,acc is %s...' % (step, loss, acc))
                # save model
                saver.save(sess, os.path.join('model/', 'mnist.ckpt'), global_step)
                summary_writer.add_summary(merge, step)
        summary_writer.close()


if __name__ == '__main__':
    train()
