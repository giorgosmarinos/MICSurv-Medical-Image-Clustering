import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import os
from glob import glob
from skimage.io import imread
import re 
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import rbf_kernel

from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot


#read images from the tif files
BASE_IMG_PATH='D:\\DSS_Visual_Analytics_XAI\\Code\\MMD-critic-master\\data_medical_images\\'
all_images_list = glob(os.path.join(BASE_IMG_PATH,'tiff_images','*.tif'))
all_images_list[:5]
print(all_images_list)

#print(os.listdir(BASE_IMG_PATH))
#print(os.path.join(BASE_IMG_PATH,'tiff_images','*.tif'))

#print(imread(all_images_list[0]).shape)
np.expand_dims(imread(all_images_list[0])[::4,::4],0).shape
jimread = lambda x: np.expand_dims(imread(x)[::2,::2],0)
test_image = jimread(all_images_list[1])
#print(test_image[0].shape)
#plt.imshow(test_image[0])

from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot


def splittraintest(X, y, testpercent):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(testpercent/100), random_state=0)
        print(X_train.shape)
        print(y_train.shape)
        print(X_test.shape)
        print(y_test.shape)

        X = X_train
        y = y_train
        testX = X_test
        testy = y_test
        
        return X_train, y_train, X_test, y_test




