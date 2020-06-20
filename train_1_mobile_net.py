import tensorflow as tf
from TimeDistributedImageDataGenerator.TimeDistributedImageDataGenerator import TimeDistributedImageDataGenerator

import common


def build_model():
    # Load MobileNet model excluding top.
    pre_model = tf.keras.applications.mobilenet.MobileNet(
        include_top=False,
        input_shape=(224, 224, 3)
    )
    pre_model.summary()
    # pre_model = tf.keras.applications.ResNet152(
    #     include_top=False,
    #     input_tensor=tf.keras.layers.Input(shape=(224, 224, 3))
    # )

    # Allow to retrain the last convolutional layer.
    trainable = 3
    for layer in pre_model.layers[:-trainable]:
        layer.trainable = False
    for layer in pre_model.layers[-trainable:]:
        layer.trainable = True

    # Build the new CNN adding a layer to flatten the convolution as required
    # to 1D for the RNN,
    cnn_model = tf.keras.models.Sequential(
        [
            pre_model,
            tf.keras.layers.GlobalMaxPool2D()
        ]
    )

    # Now build the RNN model.
    rnn_model = tf.keras.models.Sequential()

    # Process 50 frames, each of 224x244 RGB
    rnn_model.add(tf.keras.layers.TimeDistributed(cnn_model, input_shape=(5, 224, 224, 3)))
    # rnn_model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()))

    # Build the classification layer.
    rnn_model.add(tf.keras.layers.LSTM(64))
    rnn_model.add(tf.keras.layers.Dense(1024, activation='relu'))
    rnn_model.add(tf.keras.layers.Dropout(0.5))
    rnn_model.add(tf.keras.layers.Dense(512, activation='relu'))
    rnn_model.add(tf.keras.layers.Dropout(0.5))
    rnn_model.add(tf.keras.layers.Dense(256, activation='relu'))
    rnn_model.add(tf.keras.layers.Dropout(0.5))
    rnn_model.add(tf.keras.layers.Dense(128, activation='relu'))
    rnn_model.add(tf.keras.layers.Dropout(0.5))
    rnn_model.add(tf.keras.layers.Dense(1, activation='softmax'))

    return rnn_model


def train():
    model = build_model()

    checkpoint_path = 'models/1/MobileNet/ckpts/cp-{epoch:04d}.ckpt'

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path
        )
    ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=[tf.keras.metrics.Accuracy()]
    )

    train_idg = TimeDistributedImageDataGenerator(
        rotation_range=30,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        rescale=1. / 255,
        validation_split=0.2,
        time_steps=5
    )

    # train_idg = tf.keras.preprocessing.image.ImageDataGenerator(
    #     rotation_range=30,
    #     zoom_range=0.15,
    #     width_shift_range=0.2,
    #     height_shift_range=0.2,
    #     shear_range=0.15,
    #     horizontal_flip=True,
    #     rescale=1. / 255,
    #     validation_split = 0.2,
    # )

    validation_idg = tf.keras.preprocessing.image.ImageDataGenerator()

    history = model.fit(
        train_idg.flow_from_directory(
            common.TRAIN_DATA_PATH,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            shuffle=True,
            seed=common.SEED_VALUE,
            classes=['agree_pure']
            # save_to_dir='./data/train'
        ),
        # validation_data=validation_idg.flow_from_directory(
        #     common.TEST_DATA_PATH,
        #     target_size=(224, 224),
        #     batch_size=50,
        #     class_mode='categorical',
        #     shuffle=True,
        #     seed=common.SEED_VALUE,
        #     classes=['agree_pure']
        #     # save_to_dir='./data/test'
        # ),
        callbacks=callbacks,
        epochs=common.EPOCHS,
    )

    model.save('models/1/MobileNet/model/ResNet152')

    common.plot_acc_loss(history, 'models/1/MobileNet/model/plot.png')


if __name__ == '__main__':
    train()
