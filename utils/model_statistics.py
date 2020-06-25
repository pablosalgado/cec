import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.utils import layer_utils

x = []


def add_model(model):
    # Taken from model layer_utils.print_summary()
    if hasattr(model, '_collected_trainable_weights'):
        trainable_count = layer_utils.count_params(model._collected_trainable_weights)
    else:
        trainable_count = layer_utils.count_params(model.trainable_weights)

    x.append({'model': model.name, 'layers': len(model.layers), 'parameters': trainable_count})


add_model(tf.keras.applications.DenseNet121(weights=None))
add_model(tf.keras.applications.DenseNet169(weights=None))
add_model(tf.keras.applications.DenseNet201(weights=None))
add_model(tf.keras.applications.InceptionResNetV2(weights=None))
add_model(tf.keras.applications.InceptionV3(weights=None))
add_model(tf.keras.applications.MobileNet(weights=None))
add_model(tf.keras.applications.MobileNetV2(weights=None))
add_model(tf.keras.applications.NASNetLarge(weights=None))
add_model(tf.keras.applications.NASNetMobile(weights=None))
add_model(tf.keras.applications.ResNet101(weights=None))
add_model(tf.keras.applications.ResNet101V2(weights=None))
add_model(tf.keras.applications.ResNet152(weights=None))
add_model(tf.keras.applications.ResNet152V2(weights=None))
add_model(tf.keras.applications.ResNet50(weights=None))
add_model(tf.keras.applications.ResNet50V2(weights=None))
add_model(tf.keras.applications.VGG16(weights=None))
add_model(tf.keras.applications.VGG19(weights=None))
add_model(tf.keras.applications.Xception(weights=None))

df = pd.DataFrame(x)
print(df)

df.to_csv('models.csv')
