# -----------------------------------------------------------------------------
# Preprocess Large MPI Facial Expression Database.
#
# author: Pablo Salgado
# contact: pabloasalgado@gmail.com
#
# The MPI DB is provided as frame pictures instead of videos. Each picture is
# a 768x576 RBG image. This scripts extracts the face, resizes it to 224x224,
# converts it to grayscale and saves the final result.
#
# Additionally, a train, test and validation sets are created. Eight actors and
# actresses were selected for the train set, the remaining two were selected
# for testing and validations sets.
#
# This data is available at:
# https://unir-tfm-cec.s3.us-east-2.amazonaws.com/cec-data.tar.gz

import os

import cv2
import imutils.paths

import common

# Collect paths for all PNGs in the MPI directory.
image_paths = imutils.paths.list_images(common.MPI_LARGE_DB_PATH)

# For each PNG ...
for image_path in image_paths:

    # Get path parts. Last part is the file name, rest are name directories.
    path_parts = image_path.split(os.path.sep)
    filename = path_parts[-1]

    # Decide if the preprocessed image is going to training or testing sets.
    path_parts.insert(-3, 'datasets')
    path_parts[-3] = 'cec-data-48x48'

    # Build path and filename to save the preprocessed image.
    save_path = os.path.sep.join(path_parts)

    # If preprocessed image exists we don't waste our resources.
    if os.path.exists(save_path):
        continue

    # Load MPI image.
    image = cv2.imread(image_path)

    # Extract the face if possible.
    faces = common.extract_face(image)

    # If no face found log the picture path and continue.
    if len(faces) == 0:
        print(f'No face detected: {image_path}')
        continue

    # To gray scale and resize.
    face = cv2.cvtColor(faces[0], cv2.COLOR_RGB2GRAY)
    face = cv2.resize(face, (48, 48))

    # Create classname directory if necessary.
    dirs = os.path.sep.join(path_parts[:-1])
    os.makedirs(dirs, exist_ok=True)

    # Save the preprocessed image.
    cv2.imwrite(save_path, face)
    print(f'{image_path} -> {save_path}')
    faces.clear()
