# -----------------------------------------------------------------------------
# Trains a model based on MobileNet.
#
# author: Pablo Salgado
# contact: pabloasalgado@gmail.com
#
# https://unir-tfm-cec.s3.us-east-2.amazonaws.com/models/10/MobileNet.tar.gz

import os

import tensorflow as tf

import common
import generators

# Parameters
BATCH_SIZE = 8
TIME_STEPS = 128
EPOCHS = 50
MDL_PATH = 'models/10/MobileNet'

os.makedirs(MDL_PATH, exist_ok=True)

CKP_PATH = MDL_PATH + '/ckpts/cp-{epoch:04d}.ckpt'
LOG_PATH = MDL_PATH + '/training.csv'
PLT_PATH = MDL_PATH + '/plot.png'
SVD_PATH = MDL_PATH + '/model'

# Configure callbacks
CALLBACKS = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=CKP_PATH,
        save_weights_only=True,
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
        input_shape=(48, 48, 3)
    )

    # Allows to retrain the two last convolutional layers.
    for layer in pre_model.layers[:-9]:
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

    # Process n frames, each of 224x244x3
    rnn_model.add(tf.keras.layers.TimeDistributed(cnn_model, input_shape=(TIME_STEPS, 48, 48, 3)))

    # Build the classification layer.
    rnn_model.add(tf.keras.layers.LSTM(128, return_sequences=True))
    # rnn_model.add(tf.keras.layers.Dense(1024, activation='relu'))
    rnn_model.add(tf.keras.layers.Dropout(0.5))
    rnn_model.add(tf.keras.layers.LSTM(64))
    rnn_model.add(tf.keras.layers.Dropout(0.5))
    rnn_model.add(tf.keras.layers.Dense(51, activation='softmax'))

    rnn_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=['accuracy']
    )

    return rnn_model


def train():
    # Download and split data.
    common.split_data_48x48(TIME_STEPS)

    # Build and compile the model.
    model = build_model()

    # model.save_weights(CKP_PATH.format(epoch=0))

    # Load last checkpoint if any.
    # model.load_weights(
    #     tf.train.latest_checkpoint(
    #         os.path.dirname(CKP_PATH)
    #     )
    # )

    train_idg = generators.TimeDistributedImageDataGenerator(
        rotation_range=30,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        rescale=1. / 255,
        time_steps=TIME_STEPS,
    )

    validation_idg = generators.TimeDistributedImageDataGenerator(
        time_steps=TIME_STEPS,
    )

    history = model.fit(
        train_idg.flow_from_directory(
            f'{common.TRAIN_DATA_PATH}-48x48',
            target_size=(48, 48),
            batch_size=BATCH_SIZE,
            class_mode='sparse',
            shuffle=False,
            color_mode='rgb'
            # classes=['agree_pure', 'agree_considered'],
            # save_to_dir='./data/train'
        ),
        validation_data=validation_idg.flow_from_directory(
            f'{common.VALIDATION_DATA_PATH}-48x48',
            target_size=(48, 48),
            batch_size=BATCH_SIZE,
            class_mode='sparse',
            shuffle=False,
            color_mode='rgb'
            # classes=['agree_pure', 'agree_considered'],
            # save_to_dir='./data/test'
        ),
        callbacks=CALLBACKS,
        epochs=EPOCHS,
    )

    model.save(SVD_PATH)

    common.plot_acc_loss(history, PLT_PATH)


if __name__ == '__main__':
    train()