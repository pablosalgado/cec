import pathlib
import numpy as np
import tensorflow as tf
from keras_video.sliding import SlidingFrameGenerator
import common
from sklearn.metrics import classification_report

tf.config.experimental_run_functions_eagerly(True)
# noinspection SpellCheckingInspection
model = tf.keras.models.load_model('./models/trial-final/32/12/ckpts/cp-0034.ckpt')
# model.summary()

TIME_STEPS = 12
# noinspection SpellCheckingInspection
CLASSES = ['I_did_not_hear', 'I_dont_care', 'I_dont_know', 'I_dont_understand', 'agree_considered', 'agree_continue',
           'agree_pure', 'agree_reluctant', 'aha-light_bulb_moment', 'annoyed_bothered', 'annoyed_rolling-eyes',
           'arrogant', 'bored', 'compassion', 'confused', 'contempt', 'disagree_considered', 'disagree_pure',
           'disagree_reluctant', 'disbelief', 'disgust', 'embarrassment', 'fear_oops', 'fear_terror',
           'happy_achievement', 'happy_laughing', 'happy_satiated', 'happy_schadenfreude', 'imagine_negative',
           'imagine_positive', 'impressed', 'insecurity', 'not_convinced', 'pain_felt', 'pain_seen',
           'remember_negative', 'remember_positive', 'sad', 'smiling_encouraging', 'smiling_endearment',
           'smiling_flirting', 'smiling_sad-nostalgia', 'smiling_sardonic', 'smiling_triumphant', 'smiling_uncertain',
           'smiling_winning', 'smiling_yeah-right', 'thinking_considering', 'thinking_problem-solving', 'tired',
           'treudoof_bambi-eyes']

tf.keras.utils.get_file(
    fname='cec-videos-test.tar.gz',
    origin='https://unir-tfm-cec.s3.us-east-2.amazonaws.com/cec-videos-test.tar.gz',
    extract=True
)

data_aug = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet.preprocess_input
)

x = SlidingFrameGenerator(
    classes=CLASSES,
    glob_pattern=common.HOME + '/.keras/datasets/cec-videos-test/{classname}/*.avi',
    nb_frames=TIME_STEPS,
    split_val=None,
    shuffle=False,
    batch_size=1,
    target_shape=(224, 224),
    nb_channel=3,
    transformation=data_aug,
    use_frame_cache=False
)

# Extracts the actual label for each generated sequence and gets the index of each label in the CLASSES list
labels = [CLASSES.index(c) for c in [pathlib.PurePath(vi['name']).parts[-2] for vi in x.vid_info]]

# Make predictions
p = model.predict(x, verbose=1)

# Get the index of each prediction in the CLASSES list
predictions = [prediction.argmax() for prediction in p]

# Build and save the confusion matrix.
c = tf.math.confusion_matrix(labels, predictions=predictions)
np.savetxt('cfm_trial_final.csv', c.numpy(), delimiter=',')

print('\nClassification Report\n')
print(classification_report(labels, predictions, target_names=CLASSES))
