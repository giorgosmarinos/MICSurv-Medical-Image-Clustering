import pandas as pd
import numpy as np
import tensorflow as tf
import os
from glob import glob
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import ResNet101V2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from typing import Any, Dict, Iterable, Sequence, Tuple, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.metrics import concordance_index_censored
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import tensorflow.compat.v2.summary as summary
from tensorflow.python.ops import summary_ops_v2
from distutils.version import LooseVersion

from synthetic_survival_time_generation.create_risk_scores import make_risk_score_for_groups
from synthetic_survival_time_generation.generate_survival_time import SurvivalTimeGenerator

from Convolutional_NN_for_Survival_prediction.CNN_Surv import InputFunction, TrainAndEvaluateModel


assert LooseVersion(tf.__version__) >= LooseVersion("2.0.0"), \
    "This notebook requires TensorFlow 2.0 or above."

print("Using Tensorflow:", tf.__version__)

path = "D:\\DSS_Visual_Analytics_XAI\\Code\\images\\Head_CT_hemorrhage\\head_ct"

train_images_list = glob(os.path.join(path,'head_ct','*.png'))


from numpy import asarray
from keras.preprocessing.image import img_to_array
from PIL import Image 
import cv2

print(len(train_images_list))

train_images = []
labels_exl = []
labels_inc = []
for i in range(len(train_images_list)):
    #img = Image.open(train_images_list[i])
    #img.thumbnail(size=(256, 256))
    #img = img.resize((256,256))
    # convert image to numpy array
    img = cv2.imread( train_images_list[i], cv2.COLOR_BGR2RGB)
    #data = img_to_array(img)
    data = cv2.resize(img, (256, 256),interpolation = cv2.INTER_AREA)
    data=np.array(data)
    train_images.append(data)
    #if data.shape == (256,256,4):
    #    train_images.append(data)
    #    labels_inc.append(i)
    #if data.shape != (256,256,4):
    #    labels_exl.append(i)
        
print(len(train_images))

#print(train_images[0].shape)

train_images_array = np.array(train_images)


from sklearn.model_selection import train_test_split


labels = pd.read_csv('D:\\DSS_Visual_Analytics_XAI\\Code\\images\\Head_CT_hemorrhage\\labels.csv', sep=',')
labels = labels.drop(columns=["id"])

x_train, x_test, y_train, y_test = train_test_split(train_images_array, labels.values.reshape(1,-1)[0], test_size=0.45)

y = np.concatenate((y_train, y_test))

risk_score_assignment, risk_scores = make_risk_score_for_groups(y)


surv_gen = SurvivalTimeGenerator(
    num_samples=y.shape[0],
    mean_survival_time=365.,
    prob_censored=.40
)
time, event = surv_gen.gen_censored_time(risk_scores)
time_train = time[:y_train.shape[0]]
event_train = event[:y_train.shape[0]]
time_test = time[y_train.shape[0]:]
event_test = event[y_train.shape[0]:]

print("%.2f%% samples are right censored in training data." % (np.sum(~event_train) * 100. / len(event_train)))
print("%.2f%% samples are right censored in test data." % (np.sum(~event_test) * 100. / len(event_test)))


cindex = concordance_index_censored(event_test, time_test, risk_scores[y_train.shape[0]:])

print(f"Concordance index on test data with actual risk scores: {cindex[0]:.3f}")


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


train_fn = InputFunction(x_train, time_train, event_train,
                  drop_last=True,
                  shuffle=True)

eval_fn = InputFunction(x_test, time_test, event_test)

trainer = TrainAndEvaluateModel(
    model=model,
    model_dir=Path("ckpts-mnist-cnnn"),
    train_dataset=train_fn(),
    eval_dataset=eval_fn(),
    learning_rate=0.0001,
    num_epochs=300,
)

trainer.train_and_evaluate()