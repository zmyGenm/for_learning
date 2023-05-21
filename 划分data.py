# coding:utf-8

from PIL import Image
import random

f_path = r'/人工智能课程作业/任务三/train'
s_path = r'/人工智能课程作业/任务三/data'

abnormal_index = [i for i in range(300)]
normal_index = [i for i in range(300)]

random.seed(42)
random.shuffle(abnormal_index)
random.shuffle(normal_index)

#将abnormal放入train
for index,value in enumerate(abnormal_index[:int(0.8*len(abnormal_index))]):
    r_path = f_path + r'\abnormal\%s' % value + r'.png'
    save_path = s_path + r'\train\abnormal\%s' % index + r'.png'
    pic = Image.open(r_path)
    pic.save(save_path)

#将abnormal放入test
for index,value in enumerate(abnormal_index[int(0.8*len(abnormal_index)):]):
    r_path = f_path + r'\abnormal\%s' % value + r'.png'
    save_path = s_path + r'\test\abnormal\%s' % index + r'.png'
    pic = Image.open(r_path)
    pic.save(save_path)

#将normal放入train
for index,value in enumerate(normal_index[:int(0.8*len(normal_index))]):
    r_path = f_path + r'\normal\%s' % value + r'.png'
    save_path = s_path + r'\train\normal\%s' % index + r'.png'
    pic = Image.open(r_path)
    pic.save(save_path)

#将normal放入test
for index,value in enumerate(normal_index[int(0.8*len(normal_index)):]):
    r_path = f_path + r'\normal\%s' % value + r'.png'
    save_path = s_path + r'\test\normal\%s' % index + r'.png'
    pic = Image.open(r_path)
    pic.save(save_path)
