# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 17:36:54 2019

@author: hp
"""
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import model_from_json
import os
import numpy as np



json_file = open('classifier.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

from keras.preprocessing import image

test_image=image.load_img("fff.jpg",target_size=(64,64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image, axis=0)
result=loaded_model.predict(test_image)
print(f'Result {result}')
#training_set.class_indices
if result[0][0]>=0.5:
    prediction='dog'
else:
    prediction='cat'

print(prediction)
