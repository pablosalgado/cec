import os

import imutils.paths
import pandas as pd
import numpy as np

from common import MPI_LARGE_DB_PATH

image_paths = imutils.paths.list_images(MPI_LARGE_DB_PATH)

items = []
for image_path in image_paths:
    path_parts = image_path.split(os.path.sep)
    filename = path_parts[-1]

    item = [path_parts[-2], filename[0:4], filename[5:-8], filename[-7:-4]]
    items.append(item)

items = np.array(items)

df = pd.DataFrame(items)
df.columns = ['classname', 'code', 'video', 'seq']

df = df.groupby(['classname', 'code', 'video']).count()

df = df.sort_values(by='seq')

print('Shortest videos:')
print(df.head())
print()
print(f"Longest video: {df['seq'].max()}")
print(f"Shortest video: {df['seq'].min()}")
print(f"Mean frames: {df['seq'].mean()}")
