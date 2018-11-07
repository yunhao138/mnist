import tensorflow as tf
import inference
from tensorflow.examples.tutorials.mnist import input_data

def eval():
    mnist = input_data.read_data_sets('mnist/', one_hot=True)

    g=tf.Graph()
    # 这里指定运算器件为cpu，主要为了随时可以进行测试
    with g.as_default(),tf.device('/cpu:0'):
        x=tf.placeholder(tf.float32,[10000,28,28,1])
        y_=tf.placeholder(tf.float32,[10000,10])
        y=inference.lenet(x,None,False)

        correction_prediction=tf.equal(tf.argmax(y_,1),tf.argmax(y,1))
        acc=tf.reduce_mean(tf.cast(correction_prediction,tf.float32))

        saver=tf.train.Saver()

        config=tf.ConfigProto(log_device_placement=True)

    with tf.Session(graph=g,config=config) as sess:
        # sess.run(tf.global_variables_initializer())
        # 可以获取当前目录下的checkpoint文件
        ckpt=tf.train.get_checkpoint_state('model/')
        # 最新model名
        if ckpt and ckpt.model_checkpoint_path:
            validation_dict = {x: mnist.test.images.reshape(10000,28, 28, 1),
                               y_: mnist.test.labels}
            saver.restore(sess,ckpt.model_checkpoint_path)

            acc=sess.run(acc,feed_dict=validation_dict)
            print(acc)



if __name__=='__main__':
    eval()