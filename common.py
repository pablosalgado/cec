import os
import pathlib
import tensorflow as tf
import cv2
import imutils
import mtcnn
import sklearn

# Los codigos de identificación de los 10 videos
CODES = ('islf', 'kabf', 'lekf', 'milf', 'silf', 'cawm', 'chsm', 'jakm', 'juhm', 'mamm')
HOME = str(pathlib.Path.home())
MPI_LARGE_DB_PATH = f'{HOME}/.keras/large-mpi-db'
TRAIN_DATA_PATH = f'{HOME}/.keras/datasets/cec-train'
TEST_DATA_PATH = f'{HOME}/.keras/datasets/cec-test'

def extract_face(image, padding=0):
    """
    Lleva a cabo la detección y extracción de rostros de una imagen.
    :param padding: Espacio adicional a incluir en el recorte.
    :param image: Una imagen en la cual se puede o no detectar rostros.
    :return: Una lista de imagenes con los rostros recortados de la imagen dada.
    """
    detector = mtcnn.MTCNN()
    detected_faces = detector.detect_faces(image)
    faces = []
    for detected_face in detected_faces:
        x1, y1, width, height = detected_face['box']

        x1, y1 = x1 - padding, y1 - padding
        x2, y2 = x1 + width + 2 * padding, y1 + height + 2 * padding

        faces.append(image[y1:y2, x1:x2])

    return faces


def load_data():
    labels = []
    data = []

    # codes = common.CODES[0:-2]

    image_paths = imutils.paths.list_images(HOME)
    for image_path in image_paths:
        x = image_path.split(os.path.sep)
        label, file_name = x[-2], x[-1]

        # code_found = False
        # for code in codes:
        #     code_found |= code in file_name
        # if not code_found:
        #     continue

        # Cargar la siguiente imagen.
        image = cv2.imread(image_path)

        # Extraer el rostro dejando una margen de 50 píxeles para dar espacio al
        # movimiento de la cabeza de algunos videos.
        faces = extract_face(image, 50)

        if len(faces) == 0:
            print(f'No face detected: {image_path}')
            continue

        # El rostro se convierte a escala de grises para disminuir la posibilidad
        # que el extractor de características aprenda del color, el cual no es
        # relevante en la detección de la expresión facial.
        face = cv2.cvtColor(faces[0], cv2.COLOR_RGB2GRAY)
        face = cv2.resize(face, (224, 224))

        data.append(face)
        labels.append(label)

    labels = tf.keras.utils.to_categorical(labels)

    return data, labels
