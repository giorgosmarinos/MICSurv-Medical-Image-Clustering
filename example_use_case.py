import tensorflow as tf
from Convolutional_NN_for_Survival_prediction import CNN_Surv as cnn
from dataset_preparation import first_dataset 
from synthetic_survival_time_generation.create_risk_scores import make_risk_score_for_groups
import numpy as np 
from sklearn.model_selection import train_test_split
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
from keras.models import save_model, load_model
from keras.models import Model
from sklearn.cluster import KMeans
import seaborn as sns 
import matplotlib.pyplot as plt 
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_tests

def load_data():
    #read images from the tif files
    BASE_IMG_PATH='D:\\DSS_Visual_Analytics_XAI\\Code\\MMD-critic-master\\data_medical_images\\'
    all_images_list = glob(os.path.join(BASE_IMG_PATH,'tiff_images','*.tif'))
    all_images_list[:5]
    print(all_images_list)

    #print(os.listdir(BASE_IMG_PATH))
    #print(os.path.join(BASE_IMG_PATH,'tiff_images','*.tif'))

    print(imread(all_images_list[0]).shape)
    np.expand_dims(imread(all_images_list[0])[::4,::4],0).shape
    jimread = lambda x: np.expand_dims(imread(x)[::2,::2],0)
    test_image = jimread(all_images_list[1])
    #print(test_image[0].shape)
    #plt.imshow(test_image[0])

    return all_images_list, jimread


def label_extraction(all_images_list):
    check_contrast = re.compile(r'ID_([\d]+)_AGE_[\d]+_CONTRAST_([\d]+)_CT')
    label = []
    id_list = []
    for image in all_images_list:
        id_list.append(check_contrast.findall(image)[0][0])
        label.append(check_contrast.findall(image)[0][1])

    return pd.DataFrame(label,id_list)


def data_augmentation():

    images = np.stack([jimread(i) for i in all_images_list],0) 

    X_new = []
    for i in range(len(images)):
        # load the image
        img = images[i].reshape(images[i].shape[1],images[i].shape[2] )
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
        
    return np.asarray(X_new)

def label_augmentation():       
    y = label_list.values

    y_new = []
    for i in y:
        for j in range(9):
            y_new.append(i)

    return np.asarray(y_new)


def model_development():
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
    return model


def feature_extractor():
    vectors_extracted = []
    for i in range(len(x_test)):
        features = model.predict(x_test[i].reshape(1, 256, 256, 1))
        vectors_extracted.append(features)
    return vectors_extracted

def dim_reductor(TSNE=True, PCA=False):
    if TSNE==True:
        X_embedded = TSNE(n_components=2).fit_transform(vectors_reshaped)
    if PCA==True:
        pca = PCA(n_components=2, random_state=43)
        pca.fit(vectors_reshaped)
        X_embedded = pca.transform(vectors_reshaped)

    return X_embedded

def clustering():
    kmeans = KMeans(n_clusters=2,n_jobs=-1)
    kmeans.fit(X_embedded)
    holistic_table['Labels'] = pd.DataFrame(kmeans.labels_)
    return holistic_table

def visualization(holistic_table):
    table_ = pd.DataFrame(holistic_table['Labels'].values, columns = ['Labels'])

    table_.loc[table_['Labels'] == 1, 'Labels'] = 'Low Risk'
    table_.loc[table_['Labels'] == 0, 'Labels'] = 'High Risk'
    print(table_)
    ax = sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=table_['Labels'], legend='full')
    ax.axes.xaxis.set_visible(True)
    ax.axes.yaxis.set_visible(True)
    plt.show()

def kaplan_meier_curves(holistic_table):
    group_1 = holistic_table[holistic_table['Labels'] == 1]
    print(group_1)
    group_0 = holistic_table[holistic_table['Labels'] == 0]
    print(group_0)
    #group_2 = holistic_table[holistic_table['Labels'] == 2]
    #print(group_2)
    #group_3 = holistic_table[holistic_table['Labels'] == 3]
    #print(group_3)

    kmf1 = KaplanMeierFitter() 
    kmf1.fit(group_1['Time'], group_1['Events'], label="High risk group")

    kmf0 = KaplanMeierFitter() 
    kmf0.fit(group_0['Time'], group_0['Events'], label="Low risk group")

    #kmf2 = KaplanMeierFitter() 
    #kmf2.fit(group_2['Time'], group_2['Events'])

    #kmf3 = KaplanMeierFitter() 
    #kmf3.fit(group_3['Time'], group_3['Events'])

    ax = kmf1.plot()
    ax = kmf0.plot(ax = ax)
    #ax = kmf2.plot(ax = ax)
    #ax = kmf3.plot(ax = ax)


    print(group_1.shape)
    print(group_0.shape)
    #print(group_2.shape)
    #print(group_3.shape)

    #results = multivariate_logrank_test(holistic_table['Time'], holistic_table['Labels'], holistic_table['Events'])
    results = logrank_test(group_1['Time'], group_0['Time'], event_observed_A=group_1['Events'], event_observed_B=group_0['Events'])

if __name__ == "__main__":

    all_images_list, jimread = load_data()
    label_list = label_extraction(all_images_list)
    X = data_augmentation()
    y = label_augmentation()
            
    # Split train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=445, random_state=0)
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

    model = model_development()

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

    #model.save('model.h5')
    #model = load_model('C:\\Users\\geo_m\\Downloads\\model.h5')

    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

    vectors_extracted = feature_extractor()

    vectors_reshaped = np.asarray(vectors_extracted).reshape(np.asarray(vectors_extracted).shape[0],np.asarray(vectors_extracted).shape[2])

    holistic_table = pd.DataFrame(event_test, columns=['Events'])

    holistic_table['Time'] = pd.DataFrame(time_test)

    X_embedded = dim_reductor(TSNE=True)

    holistic_table = clustering()
    
    visualization(holistic_table)

    kaplan_meier_curves(holistic_table)