import os

import cv2
import imutils
from keras_preprocessing.image.affine_transformations import apply_affine_transform, flip_axis, apply_channel_shift, \
    apply_brightness_shift

import common

TRANSFORMATIONS = [
    # {
    #     'theta': 0,
    #     'tx': 0,
    #     'ty': 0,
    #     'shear': 0,
    #     'zx': 1,
    #     'zy': 1,
    #     'flip_horizontal': False,
    #     'flip_vertical': False,
    #     'channel_shift_intensity': None,
    #     'brightness': None
    # },
    # {
    #     'theta': 5,
    #     'tx': 0,
    #     'ty': 0,
    #     'shear': 0,
    #     'zx': 1,
    #     'zy': 1,
    #     'flip_horizontal': False,
    #     'flip_vertical': False,
    #     'channel_shift_intensity': None,
    #     'brightness': None
    # },
    # {
    #     'theta': -5,
    #     'tx': 0,
    #     'ty': 0,
    #     'shear': 0,
    #     'zx': 1,
    #     'zy': 1,
    #     'flip_horizontal': False,
    #     'flip_vertical': False,
    #     'channel_shift_intensity': None,
    #     'brightness': None
    # },
    # {
    #     'theta': 0,
    #     'tx': 0,
    #     'ty': 0,
    #     'shear': 0,
    #     'zx': 1,
    #     'zy': 1,
    #     'flip_horizontal': True,
    #     'flip_vertical': False,
    #     'channel_shift_intensity': None,
    #     'brightness': None
    # },
    # {
    #     'theta': 0,
    #     'tx': 100,
    #     'ty': 0,
    #     'shear': 0,
    #     'zx': 1,
    #     'zy': 1,
    #     'flip_horizontal': False,
    #     'flip_vertical': False,
    #     'channel_shift_intensity': None,
    #     'brightness': None
    # },
    # {
    #     'theta': 0,
    #     'tx': -100,
    #     'ty': 0,
    #     'shear': 0,
    #     'zx': 1,
    #     'zy': 1,
    #     'flip_horizontal': False,
    #     'flip_vertical': False,
    #     'channel_shift_intensity': None,
    #     'brightness': None
    # },
    # {
    #     'theta': 0,
    #     'tx': 0,
    #     'ty': 100,
    #     'shear': 0,
    #     'zx': 1,
    #     'zy': 1,
    #     'flip_horizontal': False,
    #     'flip_vertical': False,
    #     'channel_shift_intensity': None,
    #     'brightness': None
    # },
    # {
    #     'theta': 0,
    #     'tx': 0,
    #     'ty': -100,
    #     'shear': 0,
    #     'zx': 1,
    #     'zy': 1,
    #     'flip_horizontal': False,
    #     'flip_vertical': False,
    #     'channel_shift_intensity': None,
    #     'brightness': None
    # },
    {
        'theta': 0,
        'tx': 0,
        'ty': 0,
        'shear': 0,
        'zx': 1,
        'zy': 1,
        'flip_horizontal': False,
        'flip_vertical': False,
        'channel_shift_intensity': None,
        'brightness': 0.5
    },
    {
        'theta': 0,
        'tx': 0,
        'ty': 0,
        'shear': 0,
        'zx': 1.5,
        'zy': 1,
        'flip_horizontal': False,
        'flip_vertical': False,
        'channel_shift_intensity': None,
        'brightness': None
    },
    {
        'theta': 0,
        'tx': 0,
        'ty': 0,
        'shear': 0,
        'zx': 1,
        'zy': 1.5,
        'flip_horizontal': False,
        'flip_vertical': False,
        'channel_shift_intensity': None,
        'brightness': None
    },
]


def augment_video():
    cap = cv2.VideoCapture('/home/pablo/.keras/datasets/cec-videos/agree_considered/cawm_agree_considered.avi')
    out = cv2.VideoWriter(
        '/home/pablo/.keras/datasets/cec-videos/agree_considered/cawm_agree_considered_1.avi',
        cv2.VideoWriter_fourcc(*'XVID'),
        cap.get(cv2.CAP_PROP_FPS),
        (768, 576)
    )

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(frames):
        cap.read()
        grabbed, frame = cap.read()

        print(f'{i} -> {grabbed}')

        if not grabbed:
            continue
        continue

        if i == 0:
            cv2.imwrite('/home/pablo/.keras/datasets/cec-videos/agree_considered/original.png', frame)

        # frame = cv2.flip(frame, 0)
        # frame = cv2.resize(frame, (224, 224))
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame = img_to_array(frame)

        # frame = flip_axis(frame, 1)

        #        frame = apply_affine_transform(frame, -2)

        if i == 0:
            cv2.imwrite('/home/pablo/.keras/datasets/cec-videos/agree_considered/flip-y.png', frame)
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        out.write(frame)

    cap.release()
    out.release()

    cap = cv2.VideoCapture('/home/pablo/.keras/datasets/cec-videos/agree_considered/cawm_agree_considered_1.avi')

    return


def create_augment_videos() -> None:
    images_paths = sorted(imutils.paths.list_images(common.MPI_LARGE_DB_PATH))

    for code in common.CODES:
        for key, value in common.LABELS.items():
            filtered_images_paths = list(
                filter(
                    lambda image_path: image_path.split(os.path.sep)[-1].startswith(code)
                                       and image_path.split(os.path.sep)[-2] == key,
                    images_paths
                )
            )

            for i, transformation in enumerate(TRANSFORMATIONS):
                path = f'{common.HOME}/.keras/datasets/cec-videos-augmented/{key}'
                os.makedirs(path, exist_ok=True)
                out = cv2.VideoWriter(
                    f'{path}/{code}_{key}_{i:02d}.avi',
                    cv2.VideoWriter_fourcc(*'XVID'),
                    25,
                    (768, 576)
                )

                for image_path in filtered_images_paths:
                    x = cv2.imread(image_path)

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
                            transformation['channel_shift_intensity'],
                            2
                        )

                    if transformation.get('flip_horizontal', False):
                        x = flip_axis(x, 1)

                    if transformation.get('flip_vertical', False):
                        x = flip_axis(x, 0)

                    if transformation.get('brightness') is not None:
                        x = apply_brightness_shift(x, transformation['brightness'])
                    #
                    # x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
                    # cv2.imwrite(f'{path}/transformed.png', x)
                    out.write(x)

                out.release()

    return


if __name__ == '__main__':
    create_augment_videos()
