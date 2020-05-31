# Procesamiento de la  la base de datos "large MPI Facial Expression Database".
# Se detecta y extrae el rostro de cada imagen de la bd.
# Se convierte a escala de grises
# Se redimensiona a 224x224
# Se guarda en disco
import os

import cv2
import imutils.paths

from common import MPI_LARGE_DB_PATH, extract_face, CODES

train_codes = CODES[0:8]
test_codes = CODES[-2:]

image_paths = imutils.paths.list_images(MPI_LARGE_DB_PATH)
for image_path in image_paths:
    x = image_path.split(os.path.sep)
    filename = x[-1]
    code = filename[0:4]

    x.insert(-3, 'datasets')
    if code in train_codes:
        x[-3] = 'cec-train'
    else:
        x[-3] = 'cec-test'

    save_path = os.path.sep.join(x)

    if os.path.exists(save_path):
        continue

    image = cv2.imread(image_path)
    faces = extract_face(image, 50)

    if len(faces) == 0:
        print(f'No face detected: {image_path}')
        continue

    face = cv2.cvtColor(faces[0], cv2.COLOR_RGB2GRAY)
    face = cv2.resize(face, (224, 224))

    dirs = os.path.sep.join(x[:-1])
    os.makedirs(dirs, exist_ok=True)

    cv2.imwrite(save_path, face)
    print(f'{image_path} -> {save_path}')
    faces.clear()
