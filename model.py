import tensorflow as tf
from get_data import get_data
import numpy as np
import json
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

(train_features, train_labels), (test_features, test_labels) = get_data()

# Experimental Normalization Layers
normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_features))

def linear_model():
    model = tf.keras.Sequential([
        normalizer,
        layers.Dense(2)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss='mean_absolute_error')

    return model

def shallow_medium_model():
    model = tf.keras.Sequential([
        normalizer,
        layers.Dense(3, input_dim=4, activation='relu'),
        layers.Dense(2)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_absolute_error')

    return model

def build_medium_model():
    model = tf.keras.Sequential([
        normalizer,
        layers.Dense(12, input_dim=4, activation='relu'),
        layers.Dense(5, activation='relu'),
        layers.Dense(2),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss='mean_absolute_error')

    return model


def build_medium_model_sigmoid():
    model = tf.keras.Sequential([
        normalizer,
        layers.Dense(5, input_dim=4, activation='sigmoid'),
        layers.Dense(3, activation='sigmoid'),
        layers.Dense(2),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss='mean_absolute_error')

    return model
    

def build_deep_model():
    model = tf.keras.Sequential([
        normalizer,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(2),
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss='mean_absolute_error')

    return model
