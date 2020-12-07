from imageio import imsave, imread
from PIL import Image
import numpy as np
import tensorflow as tf
import keras
import tensorflow.keras.models
from keras.models import model_from_json
from tensorflow.python.keras.backend import set_session
from tensorflow.python.framework import ops
from keras import backend as K

tf.compat.v1.disable_v2_behavior()
session = tf.compat.v1.Session()

def init():
    # init and clear session tf keras
    init = tf.compat.v1.global_variables_initializer()
    session = tf.compat.v1.keras.backend.get_session()
    session.run(init)

    # set default graph
    graph = tf.compat.v1.get_default_graph()

    # load model
    model_softmax = load_model_softmax()

    return model_softmax, graph, session


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
    

def load_model_softmax():
    json_file = open('./model_softmax_v4.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    #load weights into new model
    loaded_model.load_weights('./model_softmax_v4.h5')

    #compile and evaluate loaded model
    loaded_model.compile(loss = 'categorical_crossentropy',
                         optimizer = 'adam', metrics = ['accuracy', f1_m, precision_m, recall_m])

    return loaded_model


def load_model_mobilenet():
    json_file = open('./model_mobilenet_v2.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    #load weights into new model
    loaded_model.load_weights('./model_moblienet_v2.h5')

    #compile and evaluate loaded model
    loaded_model.compile(loss = 'categorical_crossentropy',
                         optimizer=tf.keras.optimizers.RMSprop(lr=0.0001/10), metrics=['accuracy', f1_m, precision_m, recall_m])

    return loaded_model
