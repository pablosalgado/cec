import cv2
import numpy as np
import tensorflow as tf

tf.config.experimental_run_functions_eagerly(True)
model = tf.keras.models.load_model('./models/trial-03/32/6/model')
# model.summary()

for v in range(20):
    # cap = cv2.VideoCapture('/home/pablo/.keras/datasets/cec-videos/bored/islf_bored.avi')
    cap = cv2.VideoCapture(f'/home/pablo/.keras/datasets/cec-videos-extracted-augmented/bored/cawm_bored_{v:02}.avi')
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

    for i in range(0, int(len(frames) / 6) * 6, int(len(frames) / 6)):
        f = frames[i].reshape(1, 1, 224, 224, 3)
        x = np.append(x, f, axis=1)

    # x = tf.convert_to_tensor(x)
    p = model.predict(x, verbose=1)
    print(p)
