import os
import pathlib

import cv2
import imutils
import imutils.paths
import numpy as np
from keras_preprocessing.image.affine_transformations import apply_affine_transform, flip_axis, apply_channel_shift, \
    apply_brightness_shift

HOME = str(pathlib.Path.home())

np.random.seed(645)

# LABELS = ['bored', 'confused', 'contempt']
LABELS = [
    'agree_considered',
    'agree_continue',
    'agree_pure',
    'agree_reluctant',
    'aha-light_bulb_moment',
    'annoyed_bothered',
    'annoyed_rolling-eyes',
    'arrogant',
    'bored',
    'compassion',
    'confused',
    'contempt',
    'disagree_considered',
    'disagree_pure',
    'disagree_reluctant',
    'disbelief',
    'disgust',
    'embarrassment',
    'fear_oops',
    'fear_terror',
    'happy_achievement',
    'happy_laughing',
    'happy_satiated',
    'happy_schadenfreude',
    'I_did_not_hear',
    'I_dont_care',
    'I_dont_know',
    'I_dont_understand',
    'imagine_negative',
    'imagine_positive',
    'impressed',
    'insecurity',
    'not_convinced',
    'pain_felt',
    'pain_seen',
    'remember_negative',
    'remember_positive',
    'sad',
    'smiling_encouraging',
    'smiling_endearment',
    'smiling_flirting',
    'smiling_sad-nostalgia',
    'smiling_sardonic',
    'smiling_triumphant',
    'smiling_uncertain',
    'smiling_winning',
    'smiling_yeah-right',
    'thinking_considering',
    'thinking_problem-solving',
    'tired',
    'treudoof_bambi-eyes',
]


def get_random_transformations():
    transformations = []

    for x in range(20):
        z = np.random.uniform(.5, 1.1)
        transformations.append({
            'theta': np.random.uniform(-5, 5),
            'tx': np.random.uniform(-50, 50),
            'ty': np.random.uniform(-50, 50),
            'shear': np.random.uniform(-5, 5),
            'zx': z,
            'zy': z,
            'flip_horizontal': np.random.uniform(0, 1) > 0.5,
            'flip_vertical': False,
            'channel_shift_intensity': None,
            'brightness': np.random.uniform(0.5, 1.5),
            'grayscale': np.random.uniform(0, 1) > 0.5,
        })

    return transformations


def create_augment_videos() -> None:
    images_paths = sorted(
        imutils.paths.list_images(f'{HOME}/.keras/large-mpi-db')
    )

    for code in ['islf', 'kabf', 'lekf', 'milf', 'silf', 'cawm', 'chsm', 'jakm', 'juhm', 'mamm']:
        for label in LABELS:
            # Gets paths for all PNGs that composes the video for the current actor/actress and label.
            filtered_images_paths = list(
                filter(
                    lambda image_path: image_path.split(os.path.sep)[-1].startswith(code)
                                       and image_path.split(os.path.sep)[-2] == label,
                    images_paths
                )
            )

            # Applies 20 transformations
            for t_count, transformation in enumerate(get_random_transformations()):
                # Video output path
                path = f'{HOME}/.keras/datasets/cec-videos-augmented/{label}'
                video_path = f'{path}/{code}_{label}_{t_count:02d}.avi'

                # Checks if video has been already created
                cap = cv2.VideoCapture(video_path)
                if cap.get(cv2.CAP_PROP_FRAME_COUNT) == len(filtered_images_paths):
                    print(f'Skipping: {video_path}')
                    continue

                # Creates output dir and video output stream
                os.makedirs(path, exist_ok=True)
                out = cv2.VideoWriter(
                    video_path,
                    cv2.VideoWriter_fourcc(*'XVID'),
                    25,
                    (224, 224)
                )

                print(f'Creating: {video_path}')

                # Applies current transformation to all frames (PNGs)
                for i_count, image_path in enumerate(filtered_images_paths):
                    # Loads next frame (PNG)
                    x = cv2.imread(image_path)
                    # Resizes frame keeping aspect ratio
                    x = cv2.resize(x, (224, 168))
                    # Adds a border to fill the frame to the output size of 224x224
                    x = cv2.copyMakeBorder(x, 28, 28, 0, 0, cv2.BORDER_CONSTANT)

                    # Saves the original resized PNG for future reference
                    if i_count == 0:
                        cv2.imwrite(f'{path}/transformation_oo.png', x)

                    # Actually applies the transformation to the frame
                    x = apply_affine_transform(
                        x,
                        transformation.get('theta', 0),
                        transformation.get('tx', 0),
                        transformation.get('ty', 0),
                        transformation.get('shear', 0),
                        transformation.get('zx', 1),
                        transformation.get('zy', 1)
                    )

                    if transformation.get('channel_shift_intensity') is not None:
                        x = apply_channel_shift(
                            x,
                            transformation['channel_shift_intensity']
                        )

                    if transformation.get('flip_horizontal', False):
                        x = flip_axis(x, 1)

                    if transformation.get('flip_vertical', False):
                        x = flip_axis(x, 0)

                    if transformation.get('brightness') is not None:
                        x = apply_brightness_shift(x, transformation['brightness'])
                        x = np.uint8(x)

                    if transformation.get('grayscale', False):
                        x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
                        x = np.repeat(x[:, :, np.newaxis], 3, axis=2)

                    if i_count == 0:
                        cv2.imwrite(f'{path}/transformation_{t_count:02d}.png', x)

                    # Writes the frame to the video output stream.
                    out.write(x)

                out.release()

    return


if __name__ == '__main__':
    create_augment_videos()
