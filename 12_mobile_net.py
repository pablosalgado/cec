# -----------------------------------------------------------------------------
# Trains a model based on MobileNet.
#
# author: Pablo Salgado
# contact: pabloasalgado@gmail.com
#
# https://unir-tfm-cec.s3.us-east-2.amazonaws.com/models/10/MobileNet.tar.gz

import os

import tensorflow as tf
from keras_video import VideoFrameGenerator, SlidingFrameGenerator

import common

# Parameters
BATCH_SIZE = 32
TIME_STEPS = 12
EPOCHS = 50
MDL_PATH = 'models/12/MobileNet'

os.makedirs(MDL_PATH, exist_ok=True)

CKP_PATH = MDL_PATH + '/ckpts/cp-{epoch:04d}.ckpt'
LOG_PATH = MDL_PATH + '/training.csv'
PLT_PATH = MDL_PATH + '/plot.png'
SVD_PATH = MDL_PATH + '/model'

# Configure callbacks
CALLBACKS = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=CKP_PATH,
        # save_weights_only=True,
        verbose=1
    ),
    tf.keras.callbacks.CSVLogger(
        filename=LOG_PATH
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=2
    )
]


def build_model():
    # Load MobileNet model excluding top.
    pre_model = tf.keras.applications.mobilenet.MobileNet(
        include_top=False,
        input_shape=(224, 224, 3),
        weights='imagenet'
    )

    # Allows to retrain the last convolutional layer.
    for layer in pre_model.layers[:-3]:
        layer.trainable = False

    # Build the new CNN adding a layer to flatten the convolution as required
    # to 1D for the RNN.
    cnn_model = tf.keras.models.Sequential(
        [
            pre_model,
            tf.keras.layers.GlobalMaxPool2D()
        ]
    )

    # Now build the RNN model.
    rnn_model = tf.keras.models.Sequential()

    # Process n frames, each of 48x48x3
    rnn_model.add(tf.keras.layers.TimeDistributed(cnn_model, input_shape=(TIME_STEPS, 224, 224, 3)))

    # Build the classification layer.
    rnn_model.add(tf.keras.layers.LSTM(64))
    rnn_model.add(tf.keras.layers.Dense(1024, activation='relu'))
    rnn_model.add(tf.keras.layers.Dropout(0.5))
    rnn_model.add(tf.keras.layers.Dense(512, activation='relu'))
    rnn_model.add(tf.keras.layers.Dropout(0.5))
    rnn_model.add(tf.keras.layers.Dense(128, activation='relu'))
    rnn_model.add(tf.keras.layers.Dropout(0.5))
    rnn_model.add(tf.keras.layers.Dense(64, activation='relu'))
    rnn_model.add(tf.keras.layers.Dropout(0.5))
    rnn_model.add(tf.keras.layers.Dense(51, activation='softmax'))

    rnn_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=['accuracy']
    )

    return rnn_model


def train():
    tf.keras.utils.get_file(
        fname='cec-videos.tar.gz',
        origin='https://unir-tfm-cec.s3.us-east-2.amazonaws.com/cec-videos.tar.gz',
        extract=True
    )

    # Build and compile the model.
    model = build_model()

    # model.save_weights(CKP_PATH.format(epoch=0))

    # Load last checkpoint if any.
    # model.load_weights(
    #     tf.train.latest_checkpoint(
    #         os.path.dirname(CKP_PATH)
    #     )
    # )

    data_aug = tf.keras.preprocessing.image.ImageDataGenerator(
        zoom_range=.1,
        horizontal_flip=True,
        rotation_range=8,
        width_shift_range=.2,
        height_shift_range=.2
    )

    train_idg = SlidingFrameGenerator(
        classes=[v for k, v in common.LABELS.items()],
        glob_pattern=common.VIDEOS_PATH,
        nb_frames=TIME_STEPS,
        split=.2,
        shuffle=True,
        batch_size=BATCH_SIZE,
        target_shape=(224, 224),
        nb_channel=3,
        transformation=data_aug,
        use_frame_cache=True
    )

    validation_idg = train_idg.get_validation_generator()

    import keras_video.utils
    keras_video.utils.show_sample(train_idg)

    history = model.fit(
        train_idg,
        validation_data=validation_idg,
        callbacks=CALLBACKS,
        epochs=EPOCHS,
    )

    model.save(SVD_PATH)

    common.plot_acc_loss(history, PLT_PATH)


if __name__ == '__main__':
    train()
