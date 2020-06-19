import numpy as np
import itertools
import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as font_manager
import csv
import os


from tqdm import tqdm
from time import sleep
import time

import h5py

from sklearn.model_selection import train_test_split

min = -5
max = 5

df = pd.read_hdf('data/2DDS_data.h5', 'all')

train_range = np.concatenate((np.arange(0,160), np.arange(0,160)+200))
veri_range = np.concatenate((np.arange(160,200), np.arange(160,200)+200))

print(train_range)
print(veri_range)

#print(df.loc[train_range,:])



data_training = df.loc[train_range,:]

data_veri = df.loc[veri_range,:]


#print("test_data:\n", data_training)



fig = plt.figure()
plt.plot(data_training['x1'], data_training['x2'], 'bx')
plt.plot(data_veri['x1'], data_veri['x2'], 'ro')
plt.title("Training (80%) and Verification Data (20%)")
plt.xlabel("x1")
plt.ylabel("x2")

fig_path = "2DDS_TrainingVeri.pdf"
fig.savefig(fig_path)



data_training.to_hdf('data/2DDS_data.h5', key='training', mode='a')
data_veri.to_hdf('data/2DDS_data.h5', key='veri', mode='a')











#

# fin
