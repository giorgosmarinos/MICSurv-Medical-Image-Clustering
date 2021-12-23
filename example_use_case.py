import tensorflow as tf
from Convolutional_NN_for_Survival_prediction import CNN_Surv as cnn 
from dataset_preparation import first_dataset 
from synthetic_survival_time_generation.create_risk_scores import make_risk_score_for_groups
import numpy as np 
from dataset_preparation.first_dataset import splittraintest
from synthetic_survival_time_generation.generate_survival_time import SurvivalTimeGenerator

import pandas as pd 
from matplotlib import pyplot
import re 
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from glob import glob
import os 
from skimage.io import imread
from pathlib import Path


BASE_IMG_PATH='D:\\DSS_Visual_Analytics_XAI\\Code\\MMD-critic-master\\data_medical_images\\'
all_images_list = glob(os.path.join(BASE_IMG_PATH,'tiff_images','*.tif'))
all_images_list[:5]

print(imread(all_images_list[0]).shape)
np.expand_dims(imread(all_images_list[0])[::4,::4],0).shape
jimread = lambda x: np.expand_dims(imread(x)[::2,::2],0)
test_image = jimread(all_images_list[1])
#print(test_image[0].shape)

images = np.stack([jimread(i) for i in all_images_list],0)
print(len(images))
print(images.shape)
print(type(images))
print(images)

check_contrast = re.compile(r'ID_([\d]+)_AGE_[\d]+_CONTRAST_([\d]+)_CT')
label = []
id_list = []
for image in all_images_list:
    id_list.append(check_contrast.findall(image)[0][0])
    label.append(check_contrast.findall(image)[0][1])

label_list = pd.DataFrame(label,id_list)
print(label_list.head())

X = images 

X_new = []
for i in range(len(X)):
    # load the image
    img = X[i].reshape(X[i].shape[1],X[i].shape[2] )
    # convert to numpy array
    data = img_to_array(img)
    # expand dimension to one sample
    samples = expand_dims(data, 0)
    # create image data augmentation generator
    datagen = ImageDataGenerator(rotation_range=90)
    # prepare iterator
    it = datagen.flow(samples, batch_size=1)
    # generate samples and plot
    for i in range(9):
        # define subplot
        pyplot.subplot(330 + 1 + i)
        # generate batch of images
        batch = it.next()
        # convert to unsigned integers for viewing
        image = batch[0].astype('uint8')
        X_new.append(image)
        
        
y = label_list.values

y_new = []
for i in y:
    for j in range(9):
        y_new.append(i)
        
X = np.asarray(X_new)
y = np.asarray(y_new)


X_train, y_train, X_test, y_test = splittraintest(X, y , 30)

y = np.concatenate((y_train, y_test))

#reshape here 
print('y:',y.shape)
y = y.reshape(y.shape[0])
print('new y:', y.shape)

risk_score_assignment, risk_scores = make_risk_score_for_groups(y)


surv_gen = SurvivalTimeGenerator(
    num_samples=y.shape[0],
    mean_survival_time=365.,
    prob_censored=.45
)
time, event = surv_gen.gen_censored_time(risk_scores)
time_train = time[:y_train.shape[0]]
event_train = event[:y_train.shape[0]]
time_test = time[y_train.shape[0]:]
event_test = event[y_train.shape[0]:]



model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(6, kernel_size=(5, 5), activation='relu', name='conv_1'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(16, (5, 5), activation='relu', name='conv_2'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(120, activation='relu', name='dense_1'),
    tf.keras.layers.Dense(84, activation='relu', name='dense_2'),
    tf.keras.layers.Dense(1, activation='linear', name='dense_3')
])


train_fn = cnn.InputFunction(X_train, time_train, event_train,
                  drop_last=True,
                  shuffle=True)

eval_fn = cnn.InputFunction(X_test, time_test, event_test)

trainer = cnn.TrainAndEvaluateModel(
    model=model,
    model_dir=Path("ckpts-mnist-cnn"),
    train_dataset=train_fn(),
    eval_dataset=eval_fn(),
    learning_rate=0.0001,
    num_epochs=15,
)

trainer.train_and_evaluate()