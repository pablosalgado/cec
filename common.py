import pathlib
import mtcnn

# Los codigos de identificación de los 10 videos
CODES = ('islf', 'kabf', 'lekf', 'milf', 'silf', 'cawm', 'chsm', 'jakm', 'juhm', 'mamm')
HOME = f'{str(pathlib.Path.home())}/.keras/datasets'


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
