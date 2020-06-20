import pathlib

import dlib
import matplotlib.pyplot as plt

# User home.
HOME = str(pathlib.Path.home())

# Directory to download the Large MPI DB
MPI_LARGE_DB_PATH = f'{HOME}/.keras/large-mpi-db'

# Directory for training set.
TRAIN_DATA_PATH = f'{HOME}/.keras/datasets/cec-train'

# Directory for testing set.
TEST_DATA_PATH = f'{HOME}/.keras/datasets/cec-test'

# Directory for all preprocessed images.
ALL_DATA_PATH = f'{HOME}/.keras/datasets/cec-data'

SEED_VALUE = 436

EPOCHS = 50

# The Large MPI DB were recorded with 10 actors and actresses. These are their
# codes.
CODES = ('islf', 'kabf', 'lekf', 'milf', 'silf', 'cawm', 'chsm', 'jakm', 'juhm', 'mamm')

# The class names or categories for each facial expression.
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
