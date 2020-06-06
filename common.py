import os
import pathlib

import cv2
import dlib
import imutils.paths
import numpy as np

import matplotlib.pyplot as plt

# Los codigos de identificación de los 10 videos
CODES = ('islf', 'kabf', 'lekf', 'milf', 'silf', 'cawm', 'chsm', 'jakm', 'juhm', 'mamm')
HOME = str(pathlib.Path.home())
MPI_LARGE_DB_PATH = f'{HOME}/.keras/large-mpi-db'
TRAIN_DATA_PATH = f'{HOME}/.keras/datasets/cec-train'
TEST_DATA_PATH = f'{HOME}/.keras/datasets/cec-test'
SEED_VALUE = 436
LABELS = {
    0: 'agree_considered',
    1: 'agree_continue',
    2: 'agree_pure',
    3: 'agree_reluctant',
    4: 'aha-light_bulb_moment',
    5: 'annoyed_bothered',
    6: 'annoyed_rolling-eyes',
    7: 'arrogant',
    8: 'bored',
    9: 'compassion',
    10: 'confused',
    11: 'contempt',
    12: 'I_did_not_hear',
    13: 'disagree_considered',
    14: 'disagree_pure',
    15: 'disagree_reluctant',
    16: 'disbelief',
    17: 'disgust',
    18: 'treudoof_bambi-eyes',
    19: 'I_dont_care',
    20: 'I_dont_know',
    21: 'I_dont_understand',
    22: 'embarrassment',
    23: 'fear_oops',
    24: 'fear_terror',
    25: 'happy_achievement',
    26: 'happy_laughing',
    27: 'happy_satiated',
    28: 'happy_schadenfreude',
    29: 'imagine_negative',
    30: 'imagine_positive',
    31: 'impressed',
    32: 'insecurity',
    33: 'not_convinced',
    34: 'pain_felt',
    35: 'pain_seen',
    36: 'sad',
    37: 'smiling_yeah-right',
    38: 'smiling_encouraging',
    39: 'smiling_endearment',
    40: 'smiling_flirting',
    41: 'smiling_triumphant',
    42: 'smiling_sad-nostalgia',
    43: 'smiling_sardonic',
    44: 'smiling_uncertain',
    45: 'thinking_considering',
    46: 'thinking_problem-solving',
    47: 'remember_negative',
    48: 'remember_positive',
    49: 'tired',
    50: 'smiling_winning',
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


def load_train_data(resize_shape=(224, 224)):
    labels = []
    data = []

    reversed_labels = {v: k for k, v in LABELS.items()}

    image_paths = imutils.paths.list_images(TRAIN_DATA_PATH)
    for image_path in image_paths:
        x = image_path.split(os.path.sep)

        label = x[-2]
        labels.append(reversed_labels[label])

        image = cv2.imread(image_path)
        image = cv2.resize(image, resize_shape)
        data.append(image)

    return np.array(data), np.array(labels)


def load_test_data(resize_shape=(224, 224)):
    labels = []
    data = []

    reversed_labels = {v: k for k, v in LABELS.items()}

    image_paths = imutils.paths.list_images(TEST_DATA_PATH)
    for image_path in image_paths:
        x = image_path.split(os.path.sep)

        label = x[-2]
        labels.append(reversed_labels[label])

        image = cv2.imread(image_path)
        image = cv2.resize(image, resize_shape)
        data.append(image)

    return np.array(data), np.array(labels)


def plot_acc(history, title="Model Accuracy"):
    """Imprime una gráfica mostrando la accuracy por epoch obtenida en un entrenamiento"""
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.grid(True)


def plot_loss(history, title="Model Loss"):
    """Imprime una gráfica mostrando la pérdida por epoch obtenida en un entrenamiento"""
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.grid(True)

def plot_acc_loss(history):
  plt.figure(figsize=(15,5))
  plt.subplot(1, 2, 1)
  plot_acc(history)
  plt.subplot(1, 2, 2)
  plot_loss(history)
  plt.show()