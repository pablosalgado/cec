import tensorflow as tf
from keras_video.sliding import SlidingFrameGenerator

import common

tf.config.experimental_run_functions_eagerly(True)
model = tf.keras.models.load_model('./models/trial-final/32/12/ckpts/cp-0034.ckpt')
# model.summary()

TIME_STEPS = 12
CLASSES = ['agree_considered', 'agree_continue', 'agree_pure', 'agree_reluctant', 'aha-light_bulb_moment', 'annoyed_bothered', 'annoyed_rolling-eyes', 'arrogant', 'bored', 'compassion', 'confused', 'contempt', 'disagree_considered', 'disagree_pure', 'disagree_reluctant', 'disbelief', 'disgust', 'embarrassment', 'fear_oops', 'fear_terror', 'happy_achievement', 'happy_laughing', 'happy_satiated', 'happy_schadenfreude', 'I_did_not_hear', 'I_dont_care', 'I_dont_know', 'I_dont_understand', 'imagine_negative', 'imagine_positive', 'impressed', 'insecurity', 'not_convinced', 'pain_felt', 'pain_seen', 'remember_negative', 'remember_positive', 'sad', 'smiling_encouraging', 'smiling_endearment', 'smiling_flirting', 'smiling_sad-nostalgia', 'smiling_sardonic', 'smiling_triumphant', 'smiling_uncertain', 'smiling_winning', 'smiling_yeah-right', 'thinking_considering', 'thinking_problem-solving', 'tired', 'treudoof_bambi-eyes']

data_aug = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet.preprocess_input
)

x = SlidingFrameGenerator(
    classes=CLASSES,
    glob_pattern=common.HOME + '/.keras/datasets/cec-videos-test/{classname}/*.avi',
    nb_frames=TIME_STEPS,
    split_val=None,
    shuffle=True,
    batch_size=32,
    target_shape=(224, 224),
    nb_channel=3,
    transformation=data_aug,
    use_frame_cache=False
)

p = model.evaluate(x, verbose=1)
