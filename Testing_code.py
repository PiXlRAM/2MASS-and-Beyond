# Run this program in Google colab

from google.colab import drive
drive.mount('/content/drive')

# Importing correct directories

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import datetime
import math
from scipy import ndimage
from numpy import newaxis

import os
import cv2
import random
import cv2

from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import plot_model
from keras import callbacks
from keras.models import model_from_json

# Initial configuration for image parameters
#The number of pixels in each dimension of an image.
img_size = IMG_SIZE = 121

#Tuple with height and width of images used to reshape arrays.

CATEGORIES = ["nebulae", "galaxies_test"]
#Number of classes, one class for each of 10 digits.
num_classes = len(CATEGORIES)

#Number of colour channels for the images: 3 channel for rgb.
num_channels = 3


# Creating testing set (type <list>) from directories, and doing data augmentation by rotating images by 90, 180, 270 degrees and flipping on the x and y axes

img_shape = (121, 121, 3)


def create_dataset(DATADIR):
    dataset= []
    galaxies= 0
    nebulae= 0
    flag= 0
    count= 0
    num_augmentations= 6
      
    for category in CATEGORIES:  # do nebulae and galaxies

        path = os.path.join(DATADIR,category)  # create path to nebulae and galaxies
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=nebula 1=galaxy

        for img in os.listdir(path):  # iterate over each image per nebula and galaxy
          
                print(count)
                count= count+1
            
                img_array=cv2.imread(os.path.join(path,img))
                
                img_array=cv2.resize(img_array, dsize=(121, 121), interpolation=cv2.INTER_AREA)


                if(img_array.shape!=img_shape):
                    continue
                else:                  
                    # create all the transformations
                    
                    # 90 degrees rotation
                    img_90 = ndimage.rotate(img_array, 90)
                    
                    # 180 degrees rotation
                    img_180 = ndimage.rotate(img_array, 180)
                    
                    # 270 degrees rotation
                    img_270 = ndimage.rotate(img_array, 270)
                    
                    # flip in up-down (vertial) direction
                    img_v= np.flipud(img_array)
                    
                    # flip in left-right (horizontal) direction
                    img_h= np.fliplr(img_array)                  
                    
                    dataset.append([img_array, class_num])  # add this to our training_data
                    dataset.append([img_90, class_num])  # add 90 degrees rotation to our training_data
                    dataset.append([img_180, class_num])  # add 180 degrees rotation to our training_data
                    dataset.append([img_270, class_num])  # add 270 degrees rotation to our training_data
                    dataset.append([img_v, class_num])  # add vertical flip to our training_data
                    dataset.append([img_h, class_num])  # add horizontal flip to our training_data
                    
                    if(class_num==0): # 0 for nebula, 1 for galaxy
                        nebulae= nebulae+num_augmentations
                    else:
                        galaxies= galaxies+num_augmentations
        
        if (flag==0):
            flag= flag+1
        else:
            return dataset, nebulae, galaxies


dataset_path= "/content/drive/My Drive/Classification-ML/2MASS/dataset/"
testing_set, nebulae, galaxies=create_dataset(dataset_path)
print("DATASET SHAPE:")
print(len(testing_set))
print("GALAXY COUNT:")
print(galaxies)
print("NEBULAE COUNT:")
print(nebulae)


# Function to separate the images and labels as separate labels from the training/testing sets
def dataset_splitter(dataset):
    images= []
    labels= []
    for entry in dataset:
        images.append(entry[0])
        labels.append(entry[1])
    return images, labels
    print(labels)
    

# Creating image and label <lists>
images, labels= dataset_splitter(testing_set)

# Creating correct dimensioned array
new_images= []

i= 0
for each_image in images:
  print("i: ", i)
  tmp_image= each_image[newaxis, :, :, :]
  new_images.append(each_image)
  i += 1

new_images=np.array(new_images)

print("new_images.shape: ", new_images.shape)

new_labels = np.array(labels)


# Function to plot sample real and bogus images
def plot_images(images, cls_true, cls_pred=None):

    assert len(images) == len(cls_true)

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()
    
# Function to produce separate real and bogus training sets (for plotting)
def binary_image_splitter(images, labels):
    galaxy_im= []
    galaxy_l= []
    nebulae_im= []
    nebulae_l= []
    count= 0
    for entry in labels:
        if(entry==0):
            nebulae_im.append(images[count])
            nebulae_l.append(entry)
        else:
            galaxy_im.append(images[count])
            galaxy_l.append(entry)
        count= count+1
    return galaxy_im, galaxy_l, nebulae_im, nebulae_l


# Plotting sample galaxy and nebula images
galaxy_im, galaxy_l, nebulae_im, nebulae_l= binary_image_splitter(images, labels)
plot_images(images=nebulae_im, cls_true=nebulae_l)
plot_images(images=galaxy_im, cls_true=galaxy_l)


# load json and create model
json_file = open('/content/drive/My Drive/Classification-ML/JSON_files/model300.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("/content/drive/My Drive/Classification-ML/JSON_files/model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


results = loaded_model.evaluate(new_images, new_labels, batch_size=30, verbose=1)
loss = float(results[0])
accuracy = float(results[1])
print("Loss = " + str(loss))
print("Test Accuracy = " + str(accuracy))
scores = loaded_model.evaluate(new_images, new_labels, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], scores[1]*100))


# Predicting classes for test set
predictions= loaded_model.predict(new_images, batch_size=30, verbose=0, steps=None)
prediction_classes= (predictions>0.5) * 1

# Creating confusion matrix
length= len(prediction_classes)
labels_val_resized= np.reshape(new_labels,(length, 1)) # converting the 1D labels_test[] array to a 2D array, to make it the same shape as predictions[]
cm= confusion_matrix(labels_val_resized, prediction_classes)

# Plotting the confusion matrix and normalized confusion matrix
# taken from https://www.kaggle.com/grfiv4/plot-a-confusion-matrix

import itertools

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    

 #running the function
plot_confusion_matrix(cm = cm, normalize = False, target_names = ['Nebulae', 'Galaxy'], title = "Confusion Matrix")
plot_confusion_matrix(cm = cm, normalize = True, target_names = ['Nebulae', 'Galaxy'], title = "Confusion Matrix")


# Evaluating accuracy and loss on the test set
results = loaded_model.evaluate(new_images, new_labels, batch_size=30, verbose=1)
loss = float(results[0])
accuracy = float(results[1])
print("Loss = " + str(loss))
print("Test Accuracy = " + str(accuracy))


# Saving model in JSON format to Drive 
# taken from https://machinelearningmastery.com/save-load-keras-deep-learning-models/
from keras.models import model_from_json

# evaluate the model
scores = loaded_model.evaluate(new_images, new_labels, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], scores[1]*100))


loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("/content/drive/My Drive/Classification-ML/JSON_files/model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = model_CNN.evaluate(images_test, labels_test, verbose=0)


# Predicting probabilities for each of the values in the test set
length= len(images_test)
probs = model_CNN.predict(images_test)
probs_reshaped= np.reshape(probs,(length))

# Making an array to index probabilities in the pandas Dataframe, to then plot histograms for the confusion matrix values
index= np.arange(length)

# Plotting histograms for all four confusion matrix options (TP, TN, FP, FN)
# taken from https://github.com/DistrictDataLabs/yellowbrick/issues/749
import pandas as pd

df_predictions = pd.DataFrame({'label': labels_test, 'probs': probs_reshaped, 'index': index})

fig, axs = plt.subplots(ncols=2, nrows=2, sharex=True, sharey=True)
# show true-pos 

bins = np.arange(0, 1.01, 0.1)

def show_quarter(df, query, col, title, ax, bins, x_label=None, y_label=None, autoscale=False):
    results = df.query(query)
    results[col].hist(ax=ax, bins=bins); 
    if y_label:
        ax.set_ylabel(y_label)
    if x_label:
        ax.set_xlabel(x_label)
    ax.set_title(title + " ({})".format(results.shape[0])) #IANBOB
    if(autoscale==True):
        pass    
    else:
        pass

show_quarter(df_predictions, "label==0 and probs < 0.5", "probs", "True Negative", axs[0][0], bins, y_label="Nebulae")
show_quarter(df_predictions, "label==0 and probs >= 0.5", "probs", "False Positive", axs[0][1], bins, autoscale=True)
show_quarter(df_predictions, "label==1 and probs >= 0.5", "probs", "True Positive", axs[1][1], bins, x_label="Galaxies")
show_quarter(df_predictions, "label==1 and probs < 0.5", "probs", "False Negative", axs[1][0], bins, x_label="Nebulae", y_label="Galaxies", autoscale=True)
fig.suptitle("Probabilities per Confusion Matrix cell");


# Finding extreme outliers
query= "label==1 and probs<=0.3"
results = df_predictions.query(query)
galaxy_outliers= results['index'].values
galaxy_outliers_len= len(galaxy_outliers)

query= "label==0 and probs>=0.7"
results = df_predictions.query(query)
nebulae_outliers= results['index'].values
nebulae_outliers_len= len(nebulae_outliers)


# Plotting images of 4 extreme outliers (TN and FP)
def plot_images_outliers(images, cls_true, cls_pred=None):

    assert len(images) == len(cls_true)

    if cls_pred is None:
      fig, axes = plt.subplots(2,1)
      fig.subplots_adjust(hspace=0.3, wspace=0.3)

      for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape))

        # Show true and predicted classes.
        xlabel = "True: {0}".format(cls_true[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    else:
      fig, axes = plt.subplots(2,2)
      fig.subplots_adjust(hspace=0.3, wspace=0.3)

      for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape))

        # Show true and predicted classes.
        xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()
    
    
# Running the above plotting function
print("These images are labelled as galaxies but predicted as nebulae:")

galaxy_images= []
galaxy_labels= np.full((galaxy_outliers_len), 1)
for i in galaxy_outliers:
    img= images_test[i-1]
    galaxy_images.append(img)

plot_images_outliers(images=galaxy_images, cls_true=galaxy_labels)


print("These images are labelled as nebulae but predicted as galaxies:")

nebulae_images= []
nebulae_labels= np.full(nebulae_outliers_len, 0)  
for i in nebulae_outliers:
    img= images_test[i-1]
    nebulae_images.append(img)
    
plot_images_outliers(images=nebulae_images, cls_true=nebulae_labels)


from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import matthews_corrcoef

print('recall score: ', recall_score(labels_val_resized, prediction_classes))
print('f1 score: ', f1_score(labels_val_resized, prediction_classes))
print('Matthews Correlation Coefficient: ', matthews_corrcoef(labels_val_resized, prediction_classes))


from sklearn.metrics import roc_curve

pred=model_CNN.predict(images_test).ravel()
fpr, tpr, threshholds = roc_curve(labels_test, pred)
from sklearn.metrics import auc
auc_k=auc(fpr, tpr)
plt.figure(figsize=(20,7))
plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(auc_k))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve (zoomed in at top left)')
plt.legend(loc='best')
plt.show()
