import cv2
import glob
import numpy as np

data = np.load('./datasets/new_data.npy')
img = data[0]

mask = np.zeros(img.shape[:2], dtype='uint8')
cv2.circle(mask, (112, 112), 112, 255, -1)
img = cv2.bitwise_and(img, img, mask=mask)
cv2.imshow('img', img)
cv2.waitKey(0)