# Este script descarga la base de datos "large MPI Facial Expression Database".
# Por defecto estos archivos se descargan y descomprimen en "~/.keras/datasets"
# La función get_file de Keras sólo descarga el archivo si este no se encuentra
# en el cache.

import common
import tensorflow as tf

URL_PREFIX = 'http://www.informatik.tu-cottbus.de/gs/ZipArchiveLargeDB/1_ZipArchive-CentralCam_old-MPI-Parsing/'

for code in common.CODES:
    for m in range(1,14):
        n = common.CODES.index(code) + 1
        filename = f'MPI_large_centralcam_hi_{code}_{n:02}-{m:02}.zip'
        url = f'{URL_PREFIX}{n:02}_{code}/{filename}'

        print(url)

        tf.keras.utils.get_file(
            fname=filename,
            origin=url,
            extract=True,
            cache_subdir='large-mpi-db'
        )
