# -----------------------------------------------------------------------------
# Preprocess Large MPI Facial Expression Database.
#
# The MPI DB is provided as frame pictures instead of videos. Each picture is
# a 768x576 RBG image. This scripts extracts the face, resizes it to 224x224,
# converts it to grayscale and saves the final result.
#
# Additionally, a train set and test set are created. 8 actors and actresses
# are selected for the train set and the remaining 2 are selected for testing
# set.
#
# Two directories are created in "~/.keras/datasets": "cec-train" and "ce-test"
# 51 directories for each class are created and the preprocessed image is saved
# in them:
#
# ~/.keras/datasets/
#   cec-test/
#     agree_considered/
#       juhm_agree_considered_001.png
#       juhm_agree_considered_002.png
#       juhm_agree_considered_003.png
#     agree_continue/
#       juhm_agree_continue_001.png
#       juhm_agree_continue_002.png
#       juhm_agree_continue_003.png
#   cec-train/
#     agree_considered/
#       cawm_agree_considered_001.png
#       cawm_agree_considered_002.png
#       cawm_agree_considered_003.png
#     agree_continue/
#       cawm_agree_considered_001.png
#       cawm_agree_considered_002.png
#       cawm_agree_considered_003.png
#
# The train data are available at:
# https://unir-tfm-cec.s3.us-east-2.amazonaws.com/cec-train.tar
#
# The test data are available at:
# https://unir-tfm-cec.s3.us-east-2.amazonaws.com/cec-test.tar

import os

import cv2
import imutils.paths

from common import MPI_LARGE_DB_PATH, extract_face, CODES

# The first 8 actors and actresses are used to build the training set.
train_codes = CODES[0:8]

# The last 2 are used to build the test set.
test_codes = CODES[-2:]

# Collect paths for all PNGs in the MPI directory.
image_paths = imutils.paths.list_images(MPI_LARGE_DB_PATH)

# For each PNG ...
for image_path in image_paths:

    # Get path parts. Last part is the file name, rest are name directories.
    path_parts = image_path.split(os.path.sep)
    filename = path_parts[-1]

    # Actor/actress code is taken from first four letters of the file name:
    # juhm_agree_considered_001.png -> juhm
    code = filename[0:4]

    # Decide if the preprocessed image is going to training or testing sets.
    path_parts.insert(-3, 'datasets')
    if code in train_codes:
        path_parts[-3] = 'cec-train'
    else:
        path_parts[-3] = 'cec-test'

    # Build path and filename to save the preprocessed image.
    save_path = os.path.sep.join(path_parts)

    # If preprocessed image exists we don't waste our resources.
    if os.path.exists(save_path):
        continue

    # Load MPI image.
    image = cv2.imread(image_path)

    # Extract the face if posible.
    faces = extract_face(image, 50)

    # If no face found log the picture path and continue.
    if len(faces) == 0:
        print(f'No face detected: {image_path}')
        continue

    # To gray scale and resize.
    face = cv2.cvtColor(faces[0], cv2.COLOR_RGB2GRAY)
    face = cv2.resize(face, (224, 224))

    # Create classname directory if necessary.
    dirs = os.path.sep.join(path_parts[:-1])
    os.makedirs(dirs, exist_ok=True)

    # Save the preprocessed image.
    cv2.imwrite(save_path, face)
    print(f'{image_path} -> {save_path}')
    faces.clear()
