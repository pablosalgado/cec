# -----------------------------------------------------------------------------
# Download "Large MPI Facial Expression Database".
#
# author: Pablo Salgado
# contact: pabloasalgado@gmail.com
#
# This DB is huge, about 30GB. Although one file is available for download, this
# script instead downloads 130 compressed files.
#
# Each file is downloaded and decompressed in "~/.keras/large-mpi-db"
# Once all files are downloaded and decompressed, 51 directories are created,
# each named after the conversational expression found in each video.
#
# This DB provides pictures for the n frames of each video not the videos
# themselves, so each frame is numbered:
#
# ~/.keras/
#   large-mpi-db/
#     agree_considered/
#       cawm_agree_considered_001.png
#       cawm_agree_considered_002.png
#       cawm_agree_considered_003.png
#     agree_continue/
#       cawm_agree_continue_001.png
#       cawm_agree_continue_002.png
#       cawm_agree_continue_003.png

import common
import tensorflow as tf

URL_PREFIX = 'http://www.informatik.tu-cottbus.de/gs/ZipArchiveLargeDB/1_ZipArchive-CentralCam_old-MPI-Parsing/'

# Ten actors and actresses were recorded, each part is named after them.
for n, code in enumerate(common.CODES, start=1):
    # 13 parts are provided for each actor or actress.
    for part in range(1, 14):
        # Build filename.
        filename = f'MPI_large_centralcam_hi_{code}_{n:02}-{part:02}.zip'

        # Build download url.
        url = f'{URL_PREFIX}{n:02}_{code}/{filename}'

        print(url)

        # Let Keras take care of downloading and decompressing.
        tf.keras.utils.get_file(
            fname=filename,
            origin=url,
            extract=True,
            cache_subdir='large-mpi-db'
        )
