import tensorflow as tf
import time
import os

def create_model(OPTIMIZER, LOSSFUNCTION, no_Classes,metrics):
    LAYER = [tf.keras.layers.Flatten(input_shape=[28,28]),
             tf.keras.layers.Dense(units=300, activation='relu', name='HiddenLayer1'),
             tf.keras.layers.Dense(units=100, activation='relu', name='hiddenLayer2'),
             tf.keras.layers.Dense(units=no_Classes, activation='softmax', name='Output_Layer')]
    
    model_clf = tf.keras.Sequential(layers=LAYER)
    model_clf.compile(optimizer=OPTIMIZER, loss=LOSSFUNCTION,metrics=metrics)
    return model_clf # untrained model


def get_unique_name(filename):
    unique_file_name = time.asctime().replace(' ','_').replace(':','_')
    unique_file_name = f"{unique_file_name}_{filename}"
    return unique_file_name

def save_model(model, model_name, model_dir):
    unique_name = get_unique_name(model_name)
    model.save(model_dir + '/' + unique_name)