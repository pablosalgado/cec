import numpy as np
import tensorflow as tf

import common

x_train, y_train = common.load_train_data()
x_test, y_test = common.load_test_data()

train_data, train_labels = x_train, tf.keras.utils.to_categorical(y_train)
test_data, test_labels = x_test, tf.keras.utils.to_categorical(y_test)

np.random.seed(common.SEED_VALUE)
perm = np.random.permutation(len(x_train))
train_data, train_labels = train_data[perm], train_labels[perm]

train_augmentation = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

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
new_top = tf.keras.layers.Dense(len(train_labels[0]), activation='softmax')(new_top)

model = tf.keras.models.Model(inputs=pre_trained_model.input, outputs=new_top)
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

x = round(len(train_data) * .8)

history = model.fit(
    train_augmentation.flow(
        train_data[0:x],
        train_labels[0:x],
    ),
    epochs=5,
    validation_data=(
        train_data[x:],
        train_labels[x:]
    )
)

model.save('models/ResNet50')


