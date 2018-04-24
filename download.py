#coding:utf-8
import input_data
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
# 只显示 Error
mnist=input_data.read_data_sets("data/",one_hot="True")
