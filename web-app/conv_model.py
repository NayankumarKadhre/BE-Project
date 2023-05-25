import pickle
import cv2
from keras.models import load_model
import tensorflow as tf
import numpy as np
from labels import labels

import warnings


graph = tf.compat.v1.get_default_graph()

def prepareImg(number):
    img = cv2.imread(f'uploads/image-{number}.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28, 28))
    img = img.reshape(1, 28, 28, 1)
    print(img.shape)
    return img


def GetPredict(x):
    with graph.as_default():
        model = load_model('kaggle-ip/conv_model_Final.hdf5', compile=False)
        pred = model.predict(prepareImg(x))
        warnings.simplefilter("ignore")

        index = np.argmax(pred)
        # index -= 1
        print(labels[index])
        return labels[index]
        #chr = BE.classes_[index]
        # print(chr)
