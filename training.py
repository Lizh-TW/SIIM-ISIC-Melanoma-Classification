import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from model.cnn_model import vgg16_net, efficientnet_b1
from tensorflow.keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall

tf.random.set_seed(123)


X = np.load('./datasets/g_train_224_x.npy')
y = np.load('./datasets/g_train_224_y.npy')
X = X / 255.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=40)

model = efficientnet_b1()
model.compile(optimizer=Adam(learning_rate=0.001),
              loss=BinaryCrossentropy(label_smoothing=0.05),
              metrics=["AUC", Recall()])

history = model.fit(
    X_train, y_train, batch_size=32, epochs=30,
    verbose=2, validation_data=(X_test, y_test)
)
model.save_weights('./history/eff_b0.h5')
