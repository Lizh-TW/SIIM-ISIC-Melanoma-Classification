import tensorflow as tf
import numpy as np
from model.cnn_model import vgg16_net, efficientnet_b0
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall
import pandas as pd

out_csv = pd.read_csv('./datasets/sample_submission.csv')
test_dt = np.load('./datasets/x_test_224.npy')
model = efficientnet_b0()
model.load_weights('./history/eff_b0.h5')

model.compile(optimizer=Adam(learning_rate=0.001),
              loss=BinaryCrossentropy(label_smoothing=0.05),
              metrics=["AUC", Recall()])

prediction = model.predict(test_dt)
out_csv['target'] = prediction
out_csv.to_csv('./out.csv')