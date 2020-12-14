from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.mixture import GaussianMixture
import efficientnet.tfkeras as efn
import tensorflow as tf


def vgg16_net():
    vgg = VGG16(
        input_shape=(224, 224, 3),
        weights='imagenet',
        include_top=False
    )
    
    for layer in vgg.layers:
        layer.trainable = False

    x = Flatten()(vgg.output)
    x = Dense(512, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=vgg.input, outputs=x)

    return model


def efficientnet_b0():
    inp = tf.keras.layers.Input(shape=(224, 224, 3))
    base = efn.EfficientNetB0(
        input_shape=(224, 224, 3),
        weights='imagenet',
        include_top=False
    )
    x = base(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inp, outputs=x)
    return model
