#-*- coding:utf-8 -*-

from __future__ import print_function
from keras.models import Model,Sequential
from keras.layers import ReLU
from keras.layers import Conv2D, Input, Conv2DTranspose
import tensorflow as tf
from skimage.measure import compare_ssim
#pip install -U scikit-image
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import os, glob, sys
import numpy as np
import re
import math
import matplotlib.pyplot as plt
import cv2


def psnr(target, ref):
    # assume RGB image
    target_data = np.array(target, dtype=float)
    ref_data = np.array(ref, dtype=float)

    diff = ref_data - target_data
    diff = diff.flatten('C')

    rmse = math.sqrt(np.mean(diff ** 2.))

    return 20 * math.log10(255. / rmse)
# 
#def PSNR(y_true,y_pred):
#	return tf.image.psnr(y_true,y_pred,255)

# def tf_log10(x):
# 	numerator = tf.log(x)
# 	denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
# 	return numerator / denominator
# def PSNR(y_true, y_pred):
# 	max_pixel = 1.0
# 	return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true)))) 

def SSIM(y_true , y_pred):
	return tf.image.ssim(y_true,y_pred,255)

def predict_model(scale):
	# lrelu = LeakyReLU(alpha=0.1)
	#===========================搭建ESFNET网络=================================#
	LSRCNN = Sequential()
	
	#特征提取阶段
	LSRCNN.add(Conv2D(padding="same", kernel_size=(3, 3), kernel_initializer="he_normal", filters=64, input_shape=(None, None,1), activation="relu"))
	LSRCNN.add(Conv2D(padding="same", kernel_size=(3, 3), kernel_initializer="he_normal", activation="relu", filters=64))
	LSRCNN.add(Conv2D(padding="same", kernel_size=(3, 3), kernel_initializer="he_normal", activation="relu", filters=64))
	
	#非线性映射阶段				
	LSRCNN.add(Conv2D(padding="same", kernel_size=(3, 3), kernel_initializer="he_normal", activation="relu", filters=16))
	LSRCNN.add(Conv2D(padding="same", kernel_size=(3, 3), kernel_initializer="he_normal", activation="relu", filters=16))
	LSRCNN.add(Conv2D(padding="same", kernel_size=(3, 3), kernel_initializer="he_normal", activation="relu", filters=16))
	LSRCNN.add(Conv2D(padding="same", kernel_size=(3, 3), kernel_initializer="he_normal", activation="relu", filters=16))
	
	LSRCNN.add(Conv2D(padding="same", kernel_size=(3, 3), kernel_initializer="he_normal", activation="relu", filters=64))
	
	#反卷积重建阶段
	# LSRCNN.add(BatchNormalization())
	LSRCNN.add(Conv2DTranspose(1,(9,9),strides=(scale,scale),padding='same'))
	
	adam = Adam(lr=0.0001)
	LSRCNN.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
	
	return LSRCNN

'''
def LSRCNN(image,scale):	
		
	#===========================搭建ESFNET网络=================================#
	input_img = Input(batch_shape=(1,image.shape[1],image.shape[2],1))
	#input_img = Input(shape=(None,None,1))
	#特征提取阶段
	model = Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal')(input_img)
	model = ReLU()(model)
	model = Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal')(model)
	model = ReLU()(model)
	model = Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal')(model)

	#非线性映射阶段
	model = Conv2D(16,(3,3),padding='same',kernel_initializer='he_normal')(model)
	model = ReLU()(model)

	model = Conv2D(16,(3,3),padding='same',kernel_initializer='he_normal')(model)
	model = ReLU()(model)
	model = Conv2D(16,(3,3),padding='same',kernel_initializer='he_normal')(model)
	model = ReLU()(model)
	model = Conv2D(16,(3,3),padding='same',kernel_initializer='he_normal')(model)
	model = ReLU()(model)

	model = Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal')(model)
	model = ReLU()(model)

	#反卷积重建阶段
	model = Conv2DTranspose(1,(9,9),strides=(scale,scale),padding='same')(model)
	
	output_img = model

	model = Model(input_img, output_img)

	model.load_weights('lsrcnn_model_%d.h5' % scale)

	#pred = model.predict(x)
	return model
	'''

def evalr(data_path,scale):

	#加载测试数据集，返回数据集名称列表
	img_list = glob.glob(os.path.join(data_path,'*'))

	print(len(img_list))
	
	#循环测试每一张图片
	for img_name in img_list:
		
		IMG_GT_NAME = img_name[0:-4]+'_gt'+img_name[-4:]
		IMG_IN_NAME = img_name[0:-4]+'_in'+img_name[-4:]
		IMG_PR_NAME = img_name[0:-4]+'_pr'+img_name[-4:]
		IMG_YCrCb_NAME = img_name[0:-4]+'_ycrcb'+img_name[-4:]
		#读取原始图片 并进行裁剪imread方法读取待裁剪的图片，然后查看它的shape，输出的顺序的是高度、宽度、通道数。
		#利用数组切片的方式获取需要裁剪的图片范围。这里需要注意的是切片给出的坐标为需要裁剪的图片在原图片上的坐标，顺序为[y0:y1, x0:x1]，其中原图的左上角是坐标原点。
		
		#---------------确定ground-truth图片-----------------
		img_raw = cv2.imread(img_name,cv2.IMREAD_COLOR)
		raw_shape = img_raw.shape     #shape[0]----高   shape[1]-----宽   shape[3]-----通道
		#裁剪坐标为[y0:y1, x0:x1]
		img_crop = img_raw[0:raw_shape[0]-raw_shape[0]%scale,0:raw_shape[1]-raw_shape[1]%scale,:]
		#ground-truth图片确定
		img_ground = img_crop
		#保存ground truth图片
		cv2.imwrite(IMG_GT_NAME,img_ground)
		
		#---------------确定网络输入-------------------------
		#将图片转换到YCrCb模式
		img_crop = cv2.cvtColor(img_crop,cv2.COLOR_BGR2YCrCb)
		cv2.imwrite(IMG_YCrCb_NAME,img_crop)
		shape = img_crop.shape
		#使用cv2.resize时，参数输入是 宽×高
		y_img = cv2.resize(img_crop[:,:,0],(shape[1]//scale,shape[0]//scale),cv2.INTER_CUBIC)
		#网络输入确定
		net_input = y_img
		
		#--------------确定插值后的图片----------------------
		y_img = cv2.resize(y_img,(shape[1],shape[0]),cv2.INTER_CUBIC)
		img_crop[:,:,0] = y_img
		img_crop = cv2.cvtColor(img_crop,cv2.COLOR_YCrCb2BGR)
		img_inter = img_crop
		#保存差值后的图片
		cv2.imwrite(IMG_IN_NAME,img_inter)
		
		#--------------确定网络预测后的图片-------------------
		#先对网络输入进行预处理   网络输入格式(batch_size,height,width,channels)
		Y = np.zeros((1,net_input.shape[0],net_input.shape[1],1),dtype=float)
		Y[0,:,:,0] = net_input/255.0
		input_img = Y
		LSRCNN = predict_model(scale)
		LSRCNN.load_weights('lsrcnn_model_%d.h5' % scale)
		img_pred = LSRCNN.predict(input_img, batch_size = 1)
		#对预测后的图片进行后续处理
		img_pred = img_pred * 255.0
# 		img_pred[img_pred[:]>255] = 255
# 		img_pred[img_pred[:]<0] = 0
		img_pred = img_pred.astype(np.int8)
		img = cv2.cvtColor(img_crop,cv2.COLOR_BGR2YCrCb)
		img[:,:,0] = img_pred[0,:,:,0]
		img_pred = cv2.cvtColor(img,cv2.COLOR_YCrCb2BGR)
		#保存预测后的图片
		cv2.imwrite(IMG_PR_NAME,img_pred)
		
		#-----------------计算评估指标--------------------------
		im1 = cv2.imread(IMG_GT_NAME, cv2.IMREAD_COLOR)
		im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2YCrCb)
		im2 = cv2.imread(IMG_IN_NAME, cv2.IMREAD_COLOR)
		im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2YCrCb)
		im3 = cv2.imread(IMG_PR_NAME, cv2.IMREAD_COLOR)
		im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2YCrCb)

		print("#################PSRN###########################")
		print(img_name + "   bicubic:")
		print(cv2.PSNR(im1, im2))
		print(img_name + "   LSRCNN:")
		print(cv2.PSNR(im1, im3))
# 		print("#################SSIM###########################")
# 		(score, diff) = compare_ssim(im1, im2)		
# 		print(img_name + "   bicubic:" + " " +str(score))
# 		(score, diff) = compare_ssim(im1, im3)		
# 		print(img_name + "   LSRCNN:" + " " +str(score))

if __name__ == '__main__':
	
	evalr("data/Set1/",3)