# -----------------------------------------------------------------------------
# Trains a model based on ResNet50.
#
# author: Pablo Salgado
# contact: pabloasalgado@gmail.com
#
# https://unir-tfm-cec.s3.us-east-2.amazonaws.com/models/02/ResNet50.tar.gz

import tensorflow as tf

import common

tf.keras.utils.get_file(
    fname='cec-data.tar.gz',
    origin='https://unir-tfm-cec.s3.us-east-2.amazonaws.com/cec-data.tar.gz',
    extract=True
)

train_idg = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    rescale=1./255
)

validation_idg = tf.keras.preprocessing.image.ImageDataGenerator()

pre_trained_model = tf.keras.applications.ResNet50(
    include_top=False,
    input_tensor=tf.keras.layers.Input(shape=(224, 224, 3))
)

for layer in pre_trained_model.layers:
    layer.trainable = False

new_top = pre_trained_model.output
new_top = tf.keras.layers.AveragePooling2D(pool_size=(7, 7))(new_top)
new_top = tf.keras.layers.Flatten()(new_top)
new_top = tf.keras.layers.Dense(512, activation='relu')(new_top)
new_top = tf.keras.layers.Dropout(0.5)(new_top)
new_top = tf.keras.layers.Dense(51, activation='softmax')(new_top)

model = tf.keras.models.Model(inputs=pre_trained_model.input, outputs=new_top)
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

history = model.fit(
    train_idg.flow_from_directory(
        common.ALL_DATA_PATH,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=True,
        seed=common.SEED_VALUE,
        # classes=['agree_pure']
        # save_to_dir='./data/train'
    ),
    epochs=50
)

model.save('models/02/ResNet50')

common.plot_acc_loss(history, '../models/02/ResNet50/plot.png')
