# -----------------------------------------------------------------------------
# Trains a model based on MobileNetV2.
#
# author: Pablo Salgado
# contact: pabloasalgado@gmail.com
#
# https://unir-tfm-cec.s3.us-east-2.amazonaws.com/trial01.tar.gz

import os

import tensorflow as tf
from keras_video.sliding import SlidingFrameGenerator

import common

# Parameters
TRIAL = '01'
BATCH_SIZE = 32
TIME_STEPS = 25
EPOCHS = 500

MDL_PATH = f'models/trial{TRIAL}'
CKP_PATH = MDL_PATH + '/ckpts/cp-{epoch:04d}.ckpt'
LOG_PATH = MDL_PATH + '/training.csv'
PLT_PATH = MDL_PATH + '/plot.png'
SVD_PATH = MDL_PATH + '/model'

os.makedirs(MDL_PATH, exist_ok=True)

# Configure callbacks
CALLBACKS = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=CKP_PATH,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        verbose=1,
        patience=int(EPOCHS * .1)
    ),
    tf.keras.callbacks.CSVLogger(
        filename=LOG_PATH
    ),
    tf.keras.callbacks.TensorBoard(
        log_dir=MDL_PATH,
        histogram_freq=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        verbose=1
    ),
]


def build_model():
    # Load MobileNetV2 model excluding top.
    cnn_model = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False)

    # Allows to retrain the last convolutional layer.
    for layer in cnn_model.layers[:-3]:
        layer.trainable = False

    # Build the new CNN adding a layer to flatten the convolution as required
    # for the RNN.
    cnn_model = tf.keras.models.Sequential(
        [
            cnn_model,
            tf.keras.layers.GlobalAveragePooling2D()
        ]
    )

    # Now build the RNN model.
    rnn_model = tf.keras.models.Sequential()

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
    rnn_model.add(tf.keras.layers.Dense(3, activation='softmax'))

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

    data_aug = tf.keras.preprocessing.image.ImageDataGenerator(
        zoom_range=.1,
        horizontal_flip=True,
        rotation_range=8,
        width_shift_range=.2,
        height_shift_range=.2,
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
    )

    train_idg = SlidingFrameGenerator(
        classes=['bored', 'confused', 'contempt'],
        glob_pattern=common.VIDEOS_PATH,
        nb_frames=TIME_STEPS,
        split_val=.2,
        shuffle=True,
        batch_size=BATCH_SIZE,
        target_shape=(224, 224),
        nb_channel=3,
        transformation=data_aug,
        use_frame_cache=False
    )

    validation_idg = train_idg.get_validation_generator()

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
