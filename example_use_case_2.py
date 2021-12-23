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
from matplotlib import pyplot 

from typing import Any, Dict, Iterable, Sequence, Tuple, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.metrics import concordance_index_censored
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from Convolutional_NN_for_Survival_prediction.CNN_Surv import InputFunction, TrainAndEvaluateModel

from synthetic_survival_time_generation.create_risk_scores import make_risk_score_for_groups
from synthetic_survival_time_generation.generate_survival_time import SurvivalTimeGenerator

print("Using Tensorflow:", tf.__version__)

from distutils.version import LooseVersion

assert LooseVersion(tf.__version__) >= LooseVersion("2.0.0"), \
    "This notebook requires TensorFlow 2.0 or above."


# File Directory for both the train and test
train_path = "D:\\DSS_Visual_Analytics_XAI\\Code\\images\\Chest_CT-Scan_images_Dataset\\resized_data\\train"
test_path = "D:\\DSS_Visual_Analytics_XAI\\Code\\images\\Chest_CT-Scan_images_Dataset\\resized_data\\test"


AUTOTUNE = tf.data.experimental.AUTOTUNE

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   validation_split = 0.2,
                                  
        rotation_range=5,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        #zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')


train_dataset  = train_datagen.flow_from_directory(directory = train_path,
                                                   target_size = (256,256),
                                                   class_mode = 'categorical',
                                                  batch_size = 64)

x_train=np.concatenate([train_dataset.next()[0] for i in range(train_dataset.__len__())])
y_train=np.concatenate([train_dataset.next()[1] for i in range(train_dataset.__len__())])
print(x_train.shape)
print(y_train.shape)


AUTOTUNE = tf.data.experimental.AUTOTUNE

test_datagen = ImageDataGenerator(rescale = 1./255,
                                   validation_split = 0.2,
                                  
        rotation_range=5,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        #zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')


test_dataset  = test_datagen.flow_from_directory(directory = test_path,
                                                   target_size = (256,256),
                                                   class_mode = 'categorical',
                                                  batch_size = 64)

x_test=np.concatenate([test_dataset.next()[0] for i in range(test_dataset.__len__())])
y_test=np.concatenate([test_dataset.next()[1] for i in range(test_dataset.__len__())])
print(x_test.shape)
print(y_test.shape)


y_train = y_train.argmax(1)
y_test = y_test.argmax(1)

y = np.concatenate((y_train, y_test))

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

x_train_ = x_train[:450]
x_test_ = x_test[:200]

time_train_ = time_train[:450]
time_test_ = time_test[:200]

event_train_ = event_train[:450]
event_test_ = event_test[:200]


train_fn = InputFunction(x_train, time_train, event_train,
                  drop_last=True,
                  shuffle=True)

eval_fn = InputFunction(x_test, time_test, event_test)

trainer = TrainAndEvaluateModel(
    model=model,
    model_dir=Path("ckpts-mnist-c"),
    train_dataset=train_fn(),
    eval_dataset=eval_fn(),
    learning_rate=0.0001,
    num_epochs=15,
)

trainer.train_and_evaluate()