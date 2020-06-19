import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv


"""
The avaiblable data can be found in /Module_LoadData/data
"""

load_type = 'all'
load_type = 'training'
load_type = 'veri'  # test data

df = pd.read_hdf('data/2DDS_data.h5', load_type)

print(df)









#

# fin
