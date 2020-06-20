import tensorflow as tf

from utils.keras import generators
import common


def build_model():
    # Load MobileNet model excluding top.
    pre_model = tf.keras.applications.mobilenet.MobileNet(
        include_top=False,
        input_shape=(224, 224, 3)
    )

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

    # Process 64 frames, each of 224x244x3
    rnn_model.add(tf.keras.layers.TimeDistributed(cnn_model, input_shape=(64, 224, 224, 3)))

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
    rnn_model.add(tf.keras.layers.Dense(51, activation='softmax'))

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
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=['accuracy']
    )

    train_idg = generators.TimeDistributedImageDataGenerator(
        rotation_range=30,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        rescale=1. / 255,
        time_steps=64,
    )

    validation_idg = generators.TimeDistributedImageDataGenerator(
        time_steps=64,
    )

    history = model.fit(
        train_idg.flow_from_directory(
            common.TRAIN_DATA_PATH,
            target_size=(224, 224),
            batch_size=32,
            class_mode='sparse',
            shuffle=False,
            # classes=['agree_pure', 'agree_considered'],
            # save_to_dir='./data/train'
        ),
        validation_data=validation_idg.flow_from_directory(
            common.TEST_DATA_PATH,
            target_size=(224, 224),
            batch_size=32,
            class_mode='sparse',
            shuffle=False,
            # classes=['agree_pure', 'agree_considered'],
            # save_to_dir='./data/test'
        ),
        callbacks=callbacks,
        epochs=common.EPOCHS,
    )

    model.save('models/1/MobileNet/model/ResNet152')

    common.plot_acc_loss(history, 'models/1/MobileNet/model/plot.png')


if __name__ == '__main__':
    train()
