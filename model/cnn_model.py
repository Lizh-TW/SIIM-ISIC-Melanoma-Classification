from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model


def vgg16_net():
    vgg = VGG16(
        input_shape=(224, 224, 3),
        weights='imagenet',
        include_top=False
    )
    
    for layer in vgg.layers:
        layer.trainable = False

    x = Flatten()(vgg.output)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=vgg.input, outputs=x)

    return model
