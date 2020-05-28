import cv2
import imutils.paths
import os
import common
import tensorflow as tf


data, labels = common.load_data()

print(f'data: {len(data)}')
print(f'labels: {len(labels)}')


cv2.imshow('', data[2882])
cv2.waitKey(0)
cv2.destroyAllWindows()
