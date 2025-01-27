#coding:utf-8
import tensorflow as tf
from PIL import Image
from numpy import *
import os
import reader
import remnist
import input_data

os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
# 只显示 Error


#CNN
mnist = input_data.read_data_sets('data/', one_hot=True)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

model_dir = "model/ckpt"
model_name = "ckpt"
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

##第一层卷积
x = tf.placeholder("float", shape=[None, 784],name='x')
y_ = tf.placeholder("float", shape=[None, 10],name='y_')


W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

##第二层卷积
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

##全连接
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float",name="keep")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

##输出层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2,name = "op")

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(0.0005).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#saver = tf.train.Saver([W_conv1,b_conv1,W_conv2,b_conv2,W_fc1,b_fc1,W_fc2,b_fc2])
saver = tf.train.Saver()

with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(2000):
        batch = mnist.train.next_batch(200)
        _ , train_accuracy = sess.run([train_step,accuracy],feed_dict={x:batch[0], y_:batch[1],keep_prob:0.5})
        #if i % 100 == 0:
        if i==0:
            continue
        print("step %d, training accuracy %g" % (i, train_accuracy))

    print("Training Success!")

    saver.save(sess, os.path.join(model_dir, model_name),global_step=0)
    print("Save success！")
    sess.close()



dir_name = "./test_num"
files = os.listdir(dir_name)
cnt = len(files)
for i in range(cnt):
    print(files[i].split('_')[0])
    files[i] = dir_name + "/" + files[i]
    test_images1 = reader.pre_img(files[i])

    remnist.restore_model_ckpt(test_images1)
