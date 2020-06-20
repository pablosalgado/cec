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
# The train data are available at:
# https://unir-tfm-cec.s3.us-east-2.amazonaws.com/cec-train.tar.gz
#
# The validation data are available at:
# https://unir-tfm-cec.s3.us-east-2.amazonaws.com/cec-test.tar.gz
#
# The test data are available at:
# https://unir-tfm-cec.s3.us-east-2.amazonaws.com/cec-validation.tar.gz
#

import os

import cv2
import imutils.paths
import common
import numpy as np

train_codes = common.CODES[2:10]
test_codes = common.CODES[1:2]
validation_codes = common.CODES[0:1]

# Collect paths for all PNGs in the MPI directory.
images_paths = sorted(imutils.paths.list_images(common.MPI_LARGE_DB_PATH))

for k, v in common.LABELS.items():
    for code in common.CODES:
        filtered_images_paths = list(
            filter(
                lambda image_path: image_path.split(os.path.sep)[-1].startswith(code) and
                                   image_path.split(os.path.sep)[-2] == v,
                images_paths
            )
        )

        replace = len(filtered_images_paths) < 64
        filtered_images_ix = np.sort(
            np.random.default_rng().choice(len(filtered_images_paths), size=64, replace=replace)
        )

        for i, ix in enumerate(filtered_images_ix, start=1):
            image_path = filtered_images_paths[ix]

            # Get path parts. Last part is the file name, rest are name directories.
            path_parts = image_path.split(os.path.sep)
            filename = path_parts[-1]

            # Decide if the preprocessed image is going to training or testing sets.
            path_parts.insert(-3, 'datasets')
            if code in train_codes:
                path_parts[-3] = 'cec-train'
            elif code in test_codes:
                path_parts[-3] = 'cec-test'
            else:
                path_parts[-3] = 'cec-validation'

            # Load MPI image.
            image = cv2.imread(image_path)
            faces = common.extract_face(image, 50)

            # To gray scale and resize.
            face = cv2.cvtColor(faces[0], cv2.COLOR_RGB2GRAY)
            face = cv2.resize(face, (224, 224))

            # Create classname directory if necessary.
            dirs = os.path.sep.join(path_parts[:-1])
            os.makedirs(dirs, exist_ok=True)

            # Save the preprocessed image.
            path_parts[-1] = f'{code}_{v}_{i:03d}.png'
            save_path = os.path.sep.join(path_parts)
            cv2.imwrite(save_path, face)
            print(f'{image_path} -> {save_path}')
            faces.clear()
