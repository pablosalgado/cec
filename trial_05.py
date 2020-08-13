# -----------------------------------------------------------------------------
# Trains a model based on NASNetMobile.
#
# author: Pablo Salgado
# contact: pabloasalgado@gmail.com
#
# https://unir-tfm-cec.s3.us-east-2.amazonaws.com/trial05.tar.gz

import os

import tensorflow as tf
from keras_video.sliding import SlidingFrameGenerator

import common

# Parameters
TRIAL = '05'
BATCH_SIZE = [32]
TIME_STEPS = [12]
EPOCHS = 1000

TRL_PATH = f'models/trial-{TRIAL}'
CLASSES = ['bored', 'confused', 'contempt']


def build_model(time_steps, nout):
    # Load NASNetMobile model excluding top.
    cnn_model = tf.keras.applications.nasnet.NASNetMobile(
        include_top=False,
        input_shape=(224, 224, 3),
        weights='imagenet'
    )

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

    rnn_model.add(tf.keras.layers.TimeDistributed(cnn_model, input_shape=(time_steps, 224, 224, 3)))

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
    rnn_model.add(tf.keras.layers.Dense(nout, activation='softmax'))

    rnn_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=['accuracy']
    )

    return rnn_model


def train():
    tf.keras.utils.get_file(
        fname='cec-videos-augmented.tar.gz',
        origin='https://unir-tfm-cec.s3.us-east-2.amazonaws.com/cec-videos-augmented.tar.gz',
        extract=True
    )

    for batch_size in BATCH_SIZE:
        for time_steps in TIME_STEPS:
            path = TRL_PATH + f'/{batch_size}/{time_steps}'
            os.makedirs(path, exist_ok=True)

            # Build and compile the model.
            model = build_model(time_steps, len(CLASSES))

            data_aug = tf.keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=tf.keras.applications.nasnet.preprocess_input
            )

            train_idg = SlidingFrameGenerator(
                classes=CLASSES,
                glob_pattern=common.HOME + '/.keras/datasets/cec-videos-augmented/{classname}/*.avi',
                nb_frames=time_steps,
                split_val=.2,
                shuffle=True,
                batch_size=batch_size,
                target_shape=(224, 224),
                nb_channel=3,
                transformation=data_aug,
                use_frame_cache=False
            )

            validation_idg = train_idg.get_validation_generator()

            # Configure callbacks
            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=path + '/model',
                    monitor='val_accuracy',
                    mode='max',
                    save_best_only=True,
                    verbose=1,
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    mode='min',
                    verbose=1,
                    patience=int(EPOCHS * .01)
                ),
                tf.keras.callbacks.CSVLogger(
                    filename=path + '/log.csv'
                ),
                tf.keras.callbacks.TensorBoard(
                    log_dir=path + '/tb',
                    histogram_freq=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    verbose=1
                ),
            ]

            history = model.fit(
                train_idg,
                validation_data=validation_idg,
                callbacks=callbacks,
                epochs=EPOCHS,
            )

            common.plot_acc_loss(history, path + '/plot.png')


if __name__ == '__main__':
    train()
