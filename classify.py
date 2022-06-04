import os

import numpy as np
import pandas as pd

with open('./image_list.txt', 'r') as f_in:
  image_list = f_in.readlines()
image_list = [ os.path.basename(i.strip()) for i in image_list ]

OUT_DATA = []
for idx, image_name in enumerate(image_list):
  plane_parameter = np.load(os.path.join('./test/inference/', f'{idx}_plane_parameters_0.npy'))
  plane_mask = np.load(os.path.join('./test/inference/', f'{idx}_plane_masks_0.npy'))

  mask_area = [
    np.count_nonzero(mask)
    for mask in plane_mask
  ]
  mask_area_indexes = np.argsort(mask_area)[::-1]

  x0, x1 = None, None
  DATA = {
    'filename': image_name,
    'is_straight': 0
  }
  if len(mask_area_indexes) > 1:
    for i, idx in enumerate(mask_area_indexes[:2]):
      normalize_vector = plane_parameter[idx] / np.linalg.norm(plane_parameter[idx])
      DATA[f'x{i}'] = normalize_vector[0]
      DATA[f'y{i}'] = normalize_vector[1]
      DATA[f'z{i}'] = normalize_vector[2]
      if i == 0:
        x0 = normalize_vector
      if i == 1:
        x1 = normalize_vector
    x0_dot_x1 = DATA['x0'] * DATA['x1'] + DATA['y0'] * DATA['y1'] + DATA['z0'] * DATA['z1']
    theta = np.degrees(np.arccos(x0_dot_x1))
    DATA['x0x1_degree'] = theta
    DATA['is_straight'] = 1 if theta >= 40 else 0
  OUT_DATA.append(DATA)

pd.DataFrame(OUT_DATA).to_csv(f'planercnn.csv', index=False)
