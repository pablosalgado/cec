import os
import pathlib

import cv2
import dlib
import imutils
import imutils.paths
import numpy as np
from keras_preprocessing.image.affine_transformations import apply_affine_transform, flip_axis, apply_channel_shift, \
    apply_brightness_shift

HOME = str(pathlib.Path.home())

np.random.seed(645)

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


def extract_faces(images):
    detector = dlib.get_frontal_face_detector()
    first = True

    for image in images:
        detected_faces = detector(image)
        for detected_face in detected_faces:
            e_left = detected_face.left()
            e_top = detected_face.top()
            e_right = detected_face.right()
            e_bottom = detected_face.bottom()

            # print(f'({e_left}, {e_top}, {e_right}, {e_bottom}) -> ({e_left}, {e_top}, {e_right - e_left}, {e_bottom -e_top})')
            break

        detected_faces.clear()

        if first:
            first = False
            left = e_left
            top = e_top
            right = e_right
            bottom = e_bottom
        else:
            if e_left < left:
                left = e_left
            if e_top < top:
                top = e_top
            if e_right > right:
                right = e_right
            if e_bottom > bottom:
                bottom = e_bottom

    faces = []
    for image in images:
        faces.append(image[top:bottom, left:right])

    return faces


def extract_augment_videos():
    images_paths = sorted(
        imutils.paths.list_images(f'{HOME}/.keras/large-mpi-db')
    )

    for code in ('islf', 'kabf', 'lekf', 'milf', 'silf', 'cawm', 'chsm', 'jakm', 'juhm', 'mamm'):
        for label in LABELS:
            filtered_images_paths = list(
                filter(
                    lambda image_path: image_path.split(os.path.sep)[-1].startswith(code)
                                       and image_path.split(os.path.sep)[-2] == label,
                    images_paths
                )
            )

            for t_count, transformation in enumerate(get_random_transformations()):
                path = f'{HOME}/.keras/datasets/cec-videos-extracted-augmented/{label}'
                video_path = f'{path}/{code}_{label}_{t_count:02d}.avi'
                os.makedirs(path, exist_ok=True)
                out = cv2.VideoWriter(
                    video_path,
                    cv2.VideoWriter_fourcc(*'XVID'),
                    25,
                    (224, 224)
                )

                print(f'Creating: {video_path}')

                images = []
                for image in filtered_images_paths:
                    images.append(cv2.imread(image))

                images = extract_faces(images)

                for i_count, image in enumerate(images):
                    x = cv2.resize(image, (224, 224))
                    # x = cv2.copyMakeBorder(x, 28, 28, 0, 0, cv2.BORDER_CONSTANT)

                    if i_count == 0:
                        cv2.imwrite(f'{path}/transformation_oo.png', x)

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

                    out.write(x)

                out.release()

    return


if __name__ == '__main__':
    extract_augment_videos()
