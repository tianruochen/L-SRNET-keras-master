#-*- coding:utf-8 -*-

from __future__ import print_function
import keras
from keras import backend as K
from keras.models import Sequential,Model
from keras.layers import Dense, Activation, ReLU
from keras.layers import Conv2D, MaxPooling2D, Input, ZeroPadding2D, merge, add, Conv2DTranspose
import tensorflow as tf
from keras.models import load_model
from keras import optimizers
from keras import losses
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

import os, glob, sys, threading
import scipy.io
from scipy import ndimage, misc
import numpy as np
import re
import math

#以0.0001迭代800epochs，以0.00001迭代500epochs，以0.000001迭代200epochs.
DATA_PATH = './data/train_36/'
IMG_SIZE = (36,36,1) 
BATCH_SIZE = 128
EPOCHS = 500

#放大尺度为4
TRAIN_SCALES = [2]
VALID_SCALES = [2]
INPUT_SCALE = 2


def get_image_list(data_path, scales=[2,3,4]):

	file_list = glob.glob(os.path.join(data_path,'*'))
	print(len(file_list))
	file_list = [f for f in file_list if re.search('^\d+.mat',os.path.basename(f))]
	print(len(file_list))
	
	#train_list中存放的是训练数据对的名称[原图名，下采样的图名]
	train_list = []
	for f in file_list:
		if os.path.exists(f):
			for i in range(len(scales)):
				scale = scales[i]
				string_scale = '_'+str(scale)+'.mat'
				if os.path.exists(f[:-4]+string_scale):
					train_list.append([f,f[:-4]+string_scale])
	return train_list		


def get_image_batch(full_list,offset,scale):
	
	target_list = full_list[offset:offset+BATCH_SIZE]
	batch_x = []
	batch_y = []
	for pair in target_list:
		input_img = scipy.io.loadmat(pair[1])['patch_'+str(scale)]
		gt_img = scipy.io.loadmat(pair[0])['patch']
		batch_x.append(input_img)
		batch_y.append(gt_img)
	batch_x = np.array(batch_x)/255.0
	batch_y = np.array(batch_y)/255.0
	
	batch_x.resize([BATCH_SIZE,IMG_SIZE[0]//scale,IMG_SIZE[1]//scale,1])
	batch_y.resize([BATCH_SIZE,IMG_SIZE[0],IMG_SIZE[1],1])
	
	return batch_x, batch_y 


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

@threadsafe_generator
def image_gen(target_list,scale):
	while True:
		for step in range(len(target_list)//BATCH_SIZE):
			offset = step * BATCH_SIZE
			batch_x,batch_y = get_image_batch(target_list,offset,scale)
			yield (batch_x,batch_y)

def tf_log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

def PSNR(y_true, y_pred):
# 	max_pixel = 1.0
# 	return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true)))) 		
    return tf.image.psnr(y_true, y_pred, 1)

def SSIM(y_true , y_pred):
	return tf.image.ssim(y_true,y_pred,1)
#     y_true = np.array(y_true, dtype=np.float)
#     y_pred = np.array(y_pred, dtype=np.float)
#     u_true = np.mean(y_true)
#     u_pred = np.mean(y_pred)
#     var_true = np.var(y_true)
#     var_pred = np.var(y_pred)
#     std_true = np.sqrt(var_true)
#     std_pred = np.sqrt(var_pred)
#     c1 = np.square(0.01*7)
#     c2 = np.square(0.03*7)
    
#     u_true = tf.reduce_mean(y_true)
#     u_pred = tf.reduce_mean(y_pred)
#     var_true = tf.Variable(y_true)
#     var_pred = tf.Variable(y_pred,shape=y_pred.get_shape())
#     std_true = tf.sqrt(var_true)
#     std_pred = tf.sqrt(var_pred)
#     c1 = tf.square(0.01*7)
#     c2 = tf.square(0.03*7)
#     ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
#     denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
#     return ssim / denom
# 	

#获取训练数据和测试数据:get_image_list返回数据对列表  数据对本身也是一个列表[原图名,下采样图片名]
train_list = get_image_list('./data/train_36/', scales = TRAIN_SCALES)
test_list = get_image_list('./data/test/Set5_36/',scales = VALID_SCALES)

#===========================搭建ESFNET网络=================================#
input_img = Input(shape=(IMG_SIZE[0]//INPUT_SCALE,IMG_SIZE[1]//INPUT_SCALE,1))

#特征提取阶段
model = Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal')(input_img)
model = ReLU()(model)
model = Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal')(model)
model = ReLU()(model)
model = Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal')(model)

#非线性映射阶段
model = Conv2D(16,(1,1),padding='same',kernel_initializer='he_normal')(model)
model = ReLU()(model)

model = Conv2D(16,(3,3),padding='same',kernel_initializer='he_normal')(model)
model = ReLU()(model)
model = Conv2D(16,(3,3),padding='same',kernel_initializer='he_normal')(model)
model = ReLU()(model)
model = Conv2D(16,(3,3),padding='same',kernel_initializer='he_normal')(model)
model = ReLU()(model)

model = Conv2D(64,(1,1),padding='same',kernel_initializer='he_normal')(model)
model = ReLU()(model)

#反卷积重建阶段
model = Conv2DTranspose(1,(9,9),strides=(INPUT_SCALE,INPUT_SCALE),padding='same')(model)
#===========================网络搭建结束=====================================#

output_img = model

model = Model(input_img,output_img)

#metrics中保存的是模型训练和测试时的评价指标
model.compile(optimizer=Adam(lr=0.0001),loss='mse',metrics=[PSNR,SSIM,"accuracy"])

#model.summary()用于输出模型各层参数的状况
model.summary()

filepath = './checkpoints/weights-improvement-{epoch:04d}-{PSNR:.2f}-{SSIM:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor=[PSNR,SSIM], verbose=1,mode='max')
#lr_strategy = ReduceLROnPlateau(monitor=PSNR, factor=0.1, patience=10,  mode='auto')
callbacks_list = [checkpoint]

#model.load_weights('./checkpoints/weights-improvement-0155-64.35-0.9954.hdf5', by_name=True)
model.fit_generator(image_gen(train_list,scale=INPUT_SCALE),steps_per_epoch=len(train_list)//BATCH_SIZE,\
					validation_data=image_gen(test_list,scale=INPUT_SCALE),validation_steps=len(train_list)//BATCH_SIZE,\
					epochs=EPOCHS,workers=8,callbacks=callbacks_list)

print("Training Done!!!")
print("Saving the final model ....")

model.save_weights('lsrcnn_model_%d.h5' % INPUT_SCALE)
del model





