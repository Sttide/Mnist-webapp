import tensorflow as tf
from numpy import *
import os
import reader

def restore_model_ckpt(test_image):
    sess = tf.Session()
    saver = tf.train.import_meta_graph('./model/ckptall/ckptall-0.meta')  # 加载模型结构
    saver.restore(sess, tf.train.latest_checkpoint('./model/ckptall/'))  # 只需要指定目录就可以恢复所有变量信息


    # 获取placeholder变量
    input_x = sess.graph.get_tensor_by_name('x:0')
    # 获取需要进行计算的operator
    opt = sess.graph.get_tensor_by_name('op:0')

    #opt input_x input_keep一定要在mnist"name"属性定义好
    ret = sess.run(opt,feed_dict={input_x: test_image})
    #print(ret*10000)
    res = argmax(ret,1)
    print(res)
    return res

if __name__ == "__main__":
    dir_name="./test_num"
    files = os.listdir(dir_name)
    cnt=len(files)
    for i in range(cnt):
        print(files[i])
        #print(files[i].split('_')[0])
        files[i]=dir_name+"/"+files[i]
        test_images1=reader.pre_img(files[i])
        restore_model_ckpt(test_images1)
