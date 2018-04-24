import tensorflow as tf
from PIL import Image
from numpy import *
import os
import reader
import mnist_demo
import input_data

os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
# 只显示 Error


#CNN
mnist = input_data.read_data_sets('data/', one_hot=True)
model_dir = "model/ckptall"
model_name = "ckptall"
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

def get_weight(shape,regularizer):
    w = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    if regularizer!=None:tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b

def forward(x,regularizer):
    w1 = get_weight([784,500],regularizer)
    b1 = get_bias([500])
    y1 = tf.nn.relu(tf.matmul(x,w1)+b1)

    w2 = get_weight([500,10],regularizer)
    b2 = get_bias([10])
    y = tf.nn.softmax(tf.matmul(y1,w2)+b2,name='op')

    return y

def backward(mnist):
    regularizer = 0.0001
    x = tf.placeholder(tf.float32,[None,784],name='x')
    y_ = tf.placeholder(tf.float32,[None,10],name='y_')
    y = forward(x,regularizer)
    global_step = tf.Variable(0,trainable=False)

    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(
        0.1,global_step,mnist.train.num_examples / 200,0.99,staircase=True)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step = global_step)

    ema = tf.train.ExponentialMovingAverage(0.99, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')


    saver = tf.train.Saver()
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        for i in range(60000):
            batch = mnist.train.next_batch(200)
            _, loss_value, step,train_accuracy = sess.run([train_op,loss,global_step,accuracy],feed_dict={x:batch[0], y_:batch[1]})
            if i % 100 == 0:
                if i == 0:
                    continue
                print("step %d, loss %g, training accuracy %g" % (i, loss_value, train_accuracy))

        print("Training Success!")

        saver.save(sess, os.path.join(model_dir, model_name), global_step=0)
        print("Save success！")
        sess.close()


if __name__=="__main__":
    backward(mnist)
    dir_name = "./test_num"
    files = os.listdir(dir_name)
    cnt = len(files)
    for i in range(cnt):
        #print(files[i].split('_')[0])
        print(files[i])
        files[i] = dir_name + "/" + files[i]
        test_images1 = reader.pre_img(files[i])

        mnist_demo.restore_model_ckpt(test_images1)


