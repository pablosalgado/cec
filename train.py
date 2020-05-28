import cv2
import imutils.paths
import os
import common

image_paths = imutils.paths.list_images(common.HOME)
count = 0
for image_path in image_paths:
    count += 1
    if count < 2835:
        continue

    # Cargar la siguiente imagen.
    image = cv2.imread(image_path)

    # Extraer el rostro dejando una margen de 50 píxeles para dar espacio al
    # movimiento de la cabeza de algunos videos.
    faces = common.extract_face(image, 50)

    if len(faces) == 0:
        print(f'No face detected: {image_path}')
        continue

    # El rostro se convierte a escala de grises para disminuir la posibilidad
    # que el extractor de características aprenda del color, el cual no es
    # relevante en la detección de la expresión facial.
    face = cv2.cvtColor(faces[0], cv2.COLOR_RGB2GRAY)
    face = cv2.resize(face, (224, 224))

    label = image_path.split(os.path.sep)[-2]

    cv2.imshow('', face)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    break
