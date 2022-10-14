from distutils.sysconfig import customize_compiler
import tensorflow as tf
import pandas as pd
from api import *
import cv2

model = tf.keras.models.load_model('./model/YOLOv1.h5', custom_objects={'yolo_loss': yolo_loss})
model.summary()

test = cv2.resize(cv2.imread('./test.jpg'), (224, 224))/255.0

pred = model.predict(test.reshape(1, 224, 224, 3))
df = pd.DataFrame(pred.reshape(7, 7, 30))

print(df)