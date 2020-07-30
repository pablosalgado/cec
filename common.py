# -----------------------------------------------------------------------------
# Common routines, constants and functions used in this project.
#
# author: Pablo Salgado
# contact: pabloasalgado@gmail.com
#
import os
import pathlib

import cv2
import dlib
import imutils
import imutils.paths
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# User home.
# from mtcnn import mtcnn

HOME = str(pathlib.Path.home())

# Directory to download the Large MPI DB
MPI_LARGE_DB_PATH = f'{HOME}/.keras/large-mpi-db'

# Directory for training set.
TRAIN_DATA_PATH = f'{HOME}/.keras/datasets/cec-train'

# Directory for testing set.
TEST_DATA_PATH = f'{HOME}/.keras/datasets/cec-test'

# Directory for validation set.
VALIDATION_DATA_PATH = f'{HOME}/.keras/datasets/cec-validation'

# Directory for all preprocessed images.
ALL_DATA_PATH = f'{HOME}/.keras/datasets/cec-data'

# Videos path
VIDEOS_PATH = HOME + '/.keras/datasets/cec-videos/{classname}/*.avi'

SEED_VALUE = 436

# The Large MPI DB were recorded with 10 actors and actresses. These are their
# codes.
CODES = ('islf', 'kabf', 'lekf', 'milf', 'silf', 'cawm', 'chsm', 'jakm', 'juhm', 'mamm')

# The class names or categories for each facial expression. Key is the directory name,
# value is the name on file.
LABELS = {
    'agree_considered': 'agree_considered',
    'agree_continue': 'agree_continue',
    'agree_pure': 'agree',
    'agree_reluctant': 'agree_reluctant',
    'aha-light_bulb_moment': 'aha',
    'annoyed_bothered': 'annoyed_bothered',
    'annoyed_rolling-eyes': 'annoyed-eyeroll',
    'arrogant': 'arrogant',
    'bored': 'bored',
    'compassion': 'compassion',
    'confused': 'confused',
    'contempt': 'contempt',
    'disagree_considered': 'disagree_considered',
    'disagree_pure': 'disagree',
    'disagree_reluctant': 'disagree_reluctant',
    'disbelief': 'disbelief',
    'disgust': 'disgust',
    'embarrassment': 'embarrassment',
    'fear_oops': 'fear_oops',
    'fear_terror': 'fear_terror',
    'happy_achievement': 'happy_achievement',
    'happy_laughing': 'happy_laughing',
    'happy_satiated': 'happy_satiated',
    'happy_schadenfreude': 'schadenfreude',
    'I_did_not_hear': 'dont_hear',
    'I_dont_care': 'dont_care',
    'I_dont_know': 'dont_know',
    'I_dont_understand': 'dont_understand',
    'imagine_negative': 'imagine-negative',
    'imagine_positive': 'imagine-positive',
    'impressed': 'impressed',
    'insecurity': 'insecurity',
    'not_convinced': 'not_convinced',
    'pain_felt': 'pain_felt',
    'pain_seen': 'pain_seen',
    'remember_negative': 'remember_negative',
    'remember_positive': 'remember_positive',
    'sad': 'sad',
    'smiling_encouraging': 'smiling_encouraging',
    'smiling_endearment': 'smiling_endearment',
    'smiling_flirting': 'smiling_flirting',
    'smiling_sad-nostalgia': 'smiling_sad-nostalgia',
    'smiling_sardonic': 'smiling_sardonic',
    'smiling_triumphant': 'smiling_triumphant',
    'smiling_uncertain': 'smiling_uncertain',
    'smiling_winning': 'smiling_winning',
    'smiling_yeah-right': 'smiling_yeah-right',
    'thinking_considering': 'considering',
    'thinking_problem-solving': 'problem-solving',
    'tired': 'tired',
    'treudoof_bambi-eyes': 'treudoof',
}


def extract_face(image, padding=0):
    """
    Lleva a cabo la detección y extracción de rostros de una imagen.
    :param padding: Espacio adicional a incluir en el recorte.
    :param image: Una imagen en la cual se puede o no detectar rostros.
    :return: Una lista de imagenes con los rostros recortados de la imagen dada.
    """
    faces = []

    # detector = mtcnn.MTCNN()
    # detected_faces = detector.detect_faces(image)
    # for detected_face in detected_faces:
    #     x1, y1, width, height = detected_face['box']
    #
    #     x1, y1 = x1 - padding, y1 - padding
    #     x2, y2 = x1 + width + 2 * padding, y1 + height + 2 * padding
    #
    #     faces.append(image[y1:y2, x1:x2])

    detector = dlib.get_frontal_face_detector()
    detected_faces = detector(image)
    for detected_face in detected_faces:
        left = detected_face.left() - padding
        top = detected_face.top() - padding
        right = detected_face.right() + 2 * padding
        bottom = detected_face.bottom() + 2 * padding

        faces.append(image[top:bottom, left:right])

    detected_faces.clear()

    return faces


def plot_acc(history, title="Model Accuracy"):
    """Imprime una gráfica mostrando la accuracy por epoch obtenida en un entrenamiento"""
    plt.plot(history.history['accuracy'])

    if history.history.get('val_accuracy'):
        plt.plot(history.history['val_accuracy'])
        plt.legend(['Train', 'Val'], loc='upper left')
    else:
        plt.legend(['Train'], loc='upper left')

    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')

    plt.grid(True)


def plot_loss(history, title="Model Loss"):
    """Imprime una gráfica mostrando la pérdida por epoch obtenida en un entrenamiento"""
    plt.plot(history.history['loss'])

    if history.history.get('val_accuracy'):
        plt.plot(history.history['val_loss'])
        plt.legend(['Train', 'Val'], loc='upper right')
    else:
        plt.legend(['Train'], loc='upper right')

    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid(True)


def plot_acc_loss(history, path):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plot_acc(history)
    plt.subplot(1, 2, 2)
    plot_loss(history)
    plt.savefig(path)


def download_data():
    tf.keras.utils.get_file(
        fname='cec-data.tar.gz',
        origin='https://unir-tfm-cec.s3.us-east-2.amazonaws.com/cec-data.tar.gz',
        extract=True
    )


def download_data_48x48():
    tf.keras.utils.get_file(
        fname='cec-data-48x48.tar.gz',
        origin='https://unir-tfm-cec.s3.us-east-2.amazonaws.com/cec-data-48x48.tar.gz',
        extract=True
    )


def split_data(time_steps=8, strategy=1) -> None:
    """
    Splits large MPI DB images in three sets: training, validation and testing.
    Since RNN needs a sequence, this function randomly selects n frames from DB
    to create a sequence with the given time steps.

    Files are created in ~/.keras/datasets/cec-train, ~/.keras/datasets/cec-test
    ~/.keras/datasets/cec-validation

    :param strategy: If 1 uses a split 80%, 10%, 10% for training, validation
    and testing. If 2 uses a split of 50%, 40%, 10% for training, validation and
    testing.
    :param time_steps: Sequence step. By default 8 steps.
    :return: None
    """
    download_data()

    if strategy == 1:
        train_codes = CODES[2:10]
        test_codes = CODES[1:2]
        validation_codes = CODES[0:1]
    else:
        train_codes = CODES[5:10]
        test_codes = CODES[0:1]
        validation_codes = CODES[1:5]

    for image_path in imutils.paths.list_images(TEST_DATA_PATH):
        os.remove(image_path)
    for image_path in imutils.paths.list_images(TRAIN_DATA_PATH):
        os.remove(image_path)
    for image_path in imutils.paths.list_images(VALIDATION_DATA_PATH):
        os.remove(image_path)

        # Collect paths for all PNGs in the MPI directory.
    images_paths = sorted(imutils.paths.list_images(ALL_DATA_PATH))

    for key, value in LABELS.items():
        for code in CODES:
            filtered_images_paths = list(
                filter(
                    lambda image_path: image_path.split(os.path.sep)[-1].startswith(code)
                                       and image_path.split(os.path.sep)[-2] == key,
                    images_paths
                )
            )

            replace = len(filtered_images_paths) < time_steps
            filtered_images_ix = np.sort(
                np.random.default_rng().choice(
                    len(filtered_images_paths),
                    size=time_steps,
                    replace=replace
                )
            )

            for i, ix in enumerate(filtered_images_ix, start=1):
                image_path = filtered_images_paths[ix]

                # Get path parts. Last part is the file name, rest are name
                # directories.
                path_parts = image_path.split(os.path.sep)

                # Decide if the preprocessed image is going to training, testing
                # validation sets.
                if code in train_codes:
                    path_parts[-3] = 'cec-train'
                elif code in test_codes:
                    path_parts[-3] = 'cec-test'
                elif code in validation_codes:
                    path_parts[-3] = 'cec-validation'
                else:
                    continue

                dirs = os.path.sep.join(path_parts[:-1])
                os.makedirs(dirs, exist_ok=True)

                # Save the preprocessed image.
                image = cv2.imread(image_path)
                path_parts[-1] = f'{code}_{value}_{i:03d}.png'
                save_path = os.path.sep.join(path_parts)
                cv2.imwrite(save_path, image)
                # print(f'{image_path} -> {save_path}')


def split_data_48x48(time_steps=8) -> None:
    """
    Splits large MPI DB images in three sets: training, validation and testing.
    Since RNN needs a sequence, this function randomly selects n frames from DB
    to create a sequence with the given time steps.

    Files are created in ~/.keras/datasets/cec-train, ~/.keras/datasets/cec-test
    ~/.keras/datasets/cec-validation

    :param time_steps: Sequence step. By default 8 steps.
    :return: None
    """
    download_data_48x48()

    train_codes = CODES[2:10]
    test_codes = CODES[1:2]
    validation_codes = CODES[0:1]

    for image_path in imutils.paths.list_images(f'{TEST_DATA_PATH}-48x48'):
        os.remove(image_path)
    for image_path in imutils.paths.list_images(f'{TRAIN_DATA_PATH}-48x48'):
        os.remove(image_path)
    for image_path in imutils.paths.list_images(f'{VALIDATION_DATA_PATH}-48x48'):
        os.remove(image_path)

        # Collect paths for all PNGs in the MPI directory.
    images_paths = sorted(imutils.paths.list_images(f'{ALL_DATA_PATH}-48x48'))

    for key, value in LABELS.items():
        for code in CODES:
            filtered_images_paths = list(
                filter(
                    lambda image_path: image_path.split(os.path.sep)[-1].startswith(code)
                                       and image_path.split(os.path.sep)[-2] == key,
                    images_paths
                )
            )

            replace = len(filtered_images_paths) < time_steps
            filtered_images_ix = np.sort(
                np.random.default_rng().choice(
                    len(filtered_images_paths),
                    size=time_steps,
                    replace=replace
                )
            )

            for i, ix in enumerate(filtered_images_ix, start=1):
                image_path = filtered_images_paths[ix]

                # Get path parts. Last part is the file name, rest are name
                # directories.
                path_parts = image_path.split(os.path.sep)

                # Decide if the preprocessed image is going to training, testing
                # validation sets.
                if code in train_codes:
                    path_parts[-3] = 'cec-train-48x48'
                elif code in test_codes:
                    path_parts[-3] = 'cec-test-48x48'
                elif code in validation_codes:
                    path_parts[-3] = 'cec-validation-48x48'
                else:
                    continue

                dirs = os.path.sep.join(path_parts[:-1])
                os.makedirs(dirs, exist_ok=True)

                # Save the preprocessed image.
                image = cv2.imread(image_path)
                path_parts[-1] = f'{code}_{value}_{i:03d}.png'
                save_path = os.path.sep.join(path_parts)
                cv2.imwrite(save_path, image)
                # print(f'{image_path} -> {save_path}')
