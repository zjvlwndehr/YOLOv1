import tensorflow as tf
from glob import glob
import pandas as pd
import numpy as np
import cv2
from api import *

# GPU : 1050Ti 4GB
BATCH_SIZE = 8
EPOCH = 100
INITIALIZER = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
LEAKY_RELU = tf.keras.layers.LeakyReLU(alpha=0.1)
REGULARIZER = tf.keras.regularizers.l2(0.0005) # L2 Regularization 
OPTIMIZER = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)

IMAGE_SIZE = np.array([676, 380, 3])
CSV_PATH = 'data/train_bbox.csv'
x_train_path = './data/train/*.jpg'
INPUT_SHAPE = (224, 224, 3)
csv = load_csv(CSV_PATH)
IMAGE_PATH_LIST = ['./data/train/' + csv['image'].values[i] for i in range(0, len(csv['image'].values))]

print(IMAGE_PATH_LIST[:5])

show_bbox(csv['image'][0], cv2.imread('./data/train/' + csv['image'][0]), csv['xmin'][0], csv['ymin'][0], csv['xmax'][0], csv['ymax'][0])

################### Backbone ###################

YOLO = tf.keras.models.Sequential(name='YOLOv1')
for i in range(0, len(tf.keras.applications.VGG16(weights='imagenet', include_top=False,  input_shape=(224, 224, 3)).layers) - 1):
    YOLO.add(tf.keras.applications.VGG16(weights='imagenet', include_top=False,  input_shape=INPUT_SHAPE).layers[i])

for i in YOLO.layers:
    i.trainable = False
    if(hasattr(i, 'activation')) == True:
        i.activation = LEAKY_RELU

##################### Head #####################

# YOLO.add(tf.keras.layers.Conv2D(1024, (3, 3), activation=LEAKY_RELU, kernel_initializer=INITIALIZER, kernel_regularizer = REGULARIZER, padding = 'SAME', name = "detection_conv1", dtype='float32'))
# YOLO.add(tf.keras.layers.Conv2D(1024, (3, 3), activation=LEAKY_RELU, kernel_initializer=INITIALIZER, kernel_regularizer = REGULARIZER, padding = 'SAME', name = "detection_conv2", dtype='float32'))
# YOLO.add(tf.keras.layers.MaxPool2D((2, 2)))
# YOLO.add(tf.keras.layers.Conv2D(1024, (3, 3), activation=LEAKY_RELU, kernel_initializer=INITIALIZER, kernel_regularizer = REGULARIZER, padding = 'SAME', name = "detection_conv3", dtype='float32'))
# YOLO.add(tf.keras.layers.Conv2D(1024, (3, 3), activation=LEAKY_RELU, kernel_initializer=INITIALIZER, kernel_regularizer = REGULARIZER, padding = 'SAME', name = "detection_conv4", dtype='float32'))
# # Linear 부분
# YOLO.add(tf.keras.layers.Flatten())
# YOLO.add(tf.keras.layers.Dense(4096, activation=LEAKY_RELU, kernel_initializer = INITIALIZER, kernel_regularizer = REGULARIZER, name = "detection_linear1", dtype='float32'))
# YOLO.add(tf.keras.layers.Dropout(.5))
# # 입력값을 내보내는 출력층
# YOLO.add(tf.keras.layers.Dense(1470, kernel_initializer = INITIALIZER, kernel_regularizer = REGULARIZER, name = "detection_linear2", dtype='float32')) 
# YOLO.add(tf.keras.layers.Reshape((7, 7, 30), name = 'output', dtype='float32'))

#################### Train #####################

YOLO.summary()

# YOLO.compile(loss=yolo_loss, optimizer=OPTIMIZER, run_eagerly=True)

D_set, L_set = dataset(IMAGE_PATH_LIST, cell_labeling(csv))

# YOLO.fit(D_set, L_set, batch_size=BATCH_SIZE, epochs=EPOCH, verbose=1)

# YOLO.save('YOLOv1.h5')