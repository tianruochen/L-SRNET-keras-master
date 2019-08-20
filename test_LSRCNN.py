#-*- coding:utf-8 -*-
from keras.models import load_model
from keras.models import Sequential, Model
from keras.layers import Dense, Activation
from keras.layers import Conv2D, MaxPooling2D, Input, ZeroPadding2D, merge,Conv2DTranspose
from keras.preprocessing import image
from keras.layers.advanced_activations import PReLU,ReLU
from scipy.misc import imsave, imread, imresize, toimage
import numpy as np

import matplotlib.pyplot as plt
from keras.models import model_from_json
import cv2
from PIL import Image

INPUT_SCALE = 2
img_shape = (32, 32, 1)

model = Sequential()

img = image.load_img('./data/Set1/comic.bmp', color_mode = "grayscale")
# img = image.img_to_array(img)
# img = img[0:8,0:8,0]
#input_img = Input(shape=(IMG_SIZE[0]/INPUT_SCALE, IMG_SIZE[1]/INPUT_SCALE, 1))
input_img = Input(batch_shape=(1,img.size[1],img.size[0],1))

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

model = Model(input_img, output_img)

model.load_weights('lsrcnn_model_%d.h5' % INPUT_SCALE,by_name=True)
#model.load_weights('./checkpoints/weights-improvement-0155-64.35-0.9954.hdf5', by_name=True)
#model.load_weights('./checkpoints/weights-improvement-500-15.22.hdf5',by_name=True)
# json_string = model.to_json()  
# model = model_from_json(json_string)
#img = image.load_img('./patch.jpg', grayscale=True)
xx = image.img_to_array(img)
x = xx.astype('float32') / 255
x = np.expand_dims(x, axis=0)

pred = model.predict(x)

#test_img = np.resize(pred, (32, 32))
in_image = cv2.resize(xx, (img.size[0]*INPUT_SCALE,img.size[1]*INPUT_SCALE),cv2.INTER_CUBIC)
imsave('in_img3S.png', in_image)
test_img = np.reshape(pred, (img.size[1]*INPUT_SCALE,img.size[0]*INPUT_SCALE))
imsave('test_img3S.png', test_img)
im = Image.open('test_img3S.png')

rgb = im.convert('RGB')      
rgb.save('test_img_rgb.png')