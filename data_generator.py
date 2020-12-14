import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

tf.random.set_seed(123)


def random_microscope(image):

    do = np.random.randint(0, 10)
    image = np.array(image)
    if do==0:
        mask = np.zeros(image.shape[:2], dtype='uint8')
        cv2.circle(mask, (112, 112), 112, 255, -1)
        image = cv2.bitwise_and(image, image, mask=mask)

    return image


def parse_function(images):

    images = tf.image.random_brightness(images, 0.1)
    images = tf.image.random_saturation(images, 0.7, 1.3)
    images = tf.image.random_contrast(images, 0.6, 1.5)
    images = tf.image.random_flip_left_right(images)
    images = tf.image.random_flip_up_down(images)
    #顯微鏡擴充
    images = random_microscope(images)

    return images


if __name__ == '__main__':

    dataset = np.load('datasets/x_train_224.npy')
    new_data = np.load('./datasets/new_data.npy')
    image_target = pd.read_csv('datasets/train.csv')['target']
    x_0 = []
    x_1 = []
    size_0 = 15000
    size_1 = 1200

    for i in range(len(image_target)):

        if image_target[i] == 0:
            x_0.append(dataset[i])
        else:
            x_1.append(dataset[i])

    x_0 = x_0[:size_0]
    x_1 = np.append(x_1, new_data, axis=0)
    x_1 = list(x_1)

    while len(x_1) < size_1:
        random_int = np.random.randint(0, len(x_1))
        img = np.array(x_1[random_int])
        img = parse_function(img)
        x_1.append(img)

    g_data = np.append(x_0, x_1, axis=0)
    target = np.append(np.zeros(len(x_0)), np.ones(len(x_1)))

    print(g_data.shape)
    print(target.shape)

    np.save('./datasets/g_train_224_x', g_data)
    np.save('./datasets/g_train_224_y', target)
