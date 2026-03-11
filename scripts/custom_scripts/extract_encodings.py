
import os
import shutil
from concurrent.futures.thread import ThreadPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import bz2
from pathlib import Path

from genial.experiment.file_parsers import extract_encodings
from genial.utils.utils import extract_cont_str, convert_cont_str_to_np

data_dir = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/test22/generation_out/'

dir_list = os.listdir(data_dir)
dir_list.sort()

data_list = []

for d in dir_list:

    path = Path(f"{data_dir}{d}/hdl/mydesign_comb.v.bz2")

    # with bz2.open(path, "rt") as f:   # "rt" = read text
    #     content = f.read()

    enc = extract_encodings(path)

    data_list.append({
        'd': d,
        'enc': str(enc['input'])
    })


enc_df = pd.DataFrame(data_list)

# Derive a boolean matrix representation of the encoding sequence
encodings_input_np = (
    enc_df['enc']
    .map(lambda x: extract_cont_str(x))
    .map(lambda x: convert_cont_str_to_np(x).reshape((16, 4)).T.astype(np.bool_))
)
# Derive the flipped representation of the encoding sequence
encodings_input_np_flipped = encodings_input_np.map(lambda x: ~x)

# Sort the columns and obtain by representation for unflipped and flipped version
encodings_input_np_sorted = encodings_input_np.map(lambda x: (x[np.lexsort(x.T[::-1])]).tobytes())
encodings_input_np_flipped_sorted = encodings_input_np_flipped.map(
    lambda x: (x[np.lexsort(x.T[::-1])]).tobytes()
)

# Obtain unique representation for the flipped and unflipped version
enc_df["encodings_input_group_id"] = np.minimum(encodings_input_np_sorted, encodings_input_np_flipped_sorted)


enc_df_u = enc_df.duplicated('encodings_input_group_id').sum()

import genial.experiment.plotter as plotter

# for i in range(100):
#     fig, axes = plt.subplots(1, 1, figsize=(10, 10))
#     plotter.plot_encoding_heatmap_solo(
#         ax=axes,
#         encoding_str=str(enc_df['enc'].iloc[i]),
#         design_number="__(^^)__",
#         bitwidth=4,
#         port_type="input",
#         ax_title=f"",
#     )
#     plt.show()




enc_df.to_csv('/home/ramaudruz/data_dir/misc/encoding_check/NEW_22_enc_rec_260217.csv', index=False)

import torch
encods1 = torch.load('/home/ramaudruz/data_dir/misc/enc_pred_verif_260217/NEWW_val_encodings.pt')
preds1 = torch.load('/home/ramaudruz/data_dir/misc/enc_pred_verif_260217/NEWW_val_preds.pt')

encods2 = torch.load('/home/ramaudruz/data_dir/misc/enc_pred_verif_260217/NEWW_22_val_encodings.pt')
preds2 = torch.load('/home/ramaudruz/data_dir/misc/enc_pred_verif_260217/NEWW_22_val_preds.pt')


import matplotlib.pyplot as plt
import pandas as pd

plt.figure()

pd.Series(preds1.squeeze()).hist(bins=100, alpha=0.5, density=True)
pd.Series(preds2.squeeze()).hist(bins=100, alpha=0.5, density=True)

plt.legend(['SSL weights', 'Mid effort FT weights'])
plt.xlabel("Score using FT model 2 (ME FT)")
plt.ylabel("Density")

plt.show()



preds.min(), preds.mean(), preds.max()

encods[preds.argmin()]


fig, axes = plt.subplots(1, 1, figsize=(10, 10))
plotter.plot_encoding_heatmap_solo(
    ax=axes,
    encoding_str=str(enc_df['enc'].iloc[767]),
    design_number="__(^^)__",
    bitwidth=4,
    port_type="input",
    ax_title=f"",
)
plt.show()

