import cv2
import numpy as np
import tensorflow as tf
from keras_video.sliding import SlidingFrameGenerator

import common

tf.config.experimental_run_functions_eagerly(True)
model = tf.keras.models.load_model('./models/trial-07/32/12/model')
# model.summary()

TIME_STEPS = 12
CLASSES = ['bored']

for v in range(20):
    # cap = cv2.VideoCapture('/home/pablo/.keras/datasets/cec-videos/bored/islf_bored.avi')
    cap = cv2.VideoCapture(f'/home/pablo/.keras/datasets/cec-videos-augmented/bored/cawm_bored_{v:02}.avi')
    more = True
    frames = []
    while more:
        more, frame = cap.read()

        if more:
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    cap.release()

    x = np.zeros((1, 0, 224, 224, 3), dtype=int)

    for i in range(0, int(len(frames) / TIME_STEPS) * TIME_STEPS, int(len(frames) / TIME_STEPS)):
        f = frames[i].reshape(1, 1, 224, 224, 3)
        x = np.append(x, f, axis=1)

    data_aug = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet.preprocess_input
    )

    x = SlidingFrameGenerator(
        classes=CLASSES,
        glob_pattern=common.HOME + '/.keras/datasets/cec-videos-test/{classname}/*.avi',
        nb_frames=TIME_STEPS,
        split_val=None,
        shuffle=True,
        batch_size=32,
        target_shape=(224, 224),
        nb_channel=3,
        transformation=data_aug,
        use_frame_cache=False
    )

    # x = tf.convert_to_tensor(x)
    p = model.predict(x, verbose=1)
    print(p.argmax())
