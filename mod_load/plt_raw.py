import numpy as np
import itertools
import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as font_manager
import csv
import os
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap  # allows the creation of a custom cmap

from tqdm import tqdm
from time import sleep
import time

import h5py

#from Module_Analysis.Set_Load_meta import LoadMetaData

# # Add path
import sys, os
file_dir = os.path.dirname(__file__)
currentDirectory = os.getcwd()
print(">> ", currentDirectory.replace('\mod_load', ''))
sys.path.append(currentDirectory.replace('\mod_load', ''))

# # set the working directory
os.chdir(currentDirectory.replace('\mod_load', ''))

from LoadData import Load_Data




matplotlib .rcParams['font.family'] = 'Arial'  # 'serif'
matplotlib .rcParams['font.size'] = 8  # tixks and title
matplotlib .rcParams['figure.titlesize'] = 'medium'
matplotlib .rcParams['axes.labelsize'] = 10  # axis labels
matplotlib .rcParams['axes.linewidth'] = 1  # box edge
#matplotlib .rcParams['mathtext.fontset'] = 'Arial'  # 'cm'
matplotlib.rc('pdf', fonttype=42)  # embeds the font, so can import to inkscape
matplotlib .rcParams["legend.labelspacing"] = 0.25

#matplotlib .rcParams['lines.linewidth'] = 0.85
matplotlib .rcParams['lines.markersize'] = 3.5
matplotlib .rcParams['lines.markeredgewidth'] = 0.5



matplotlib .rcParams["figure.figsize"] = [3.1,2.7]
matplotlib .rcParams["figure.autolayout"] = True
# ********************************************************************

train_ref = 'na'
veri_ref = 'na'

set = '2DDS'
train_ref = 'a'
veri_ref = 'b'



TrimDict = {}
TrimDict['DE'] = {}
TrimDict['DE']['TestVerify'] = 1
TrimDict['DE']['UseCustom_NewAttributeData'] = 0
TrimDict['DE']['training_data'] = set
TrimDict['DE']['FitScheme'] = 'error'

TrimDict['spice'] = {}
TrimDict['spice']['Vmin'] = -5
TrimDict['spice']['Vmax'] = 5


"""if set == '2DDS' or set == 'flipped_2DDS':
    the_axis = [0.5, 3.5, 0.5, 3.5]  # 2DDS
elif set == 'con2DDS':
    the_axis = [-4, 4, -4, 4]  # con2DDS"""


mks = 3.5
mks_star = mks + 1.5






# # # # plot responce data # # # #
fig = plt.figure(1)

plt.xlabel('$a_1$')
plt.ylabel('$a_2$')



# #  plot training data too
#data = 'all'
data = 'train'
train_data_X, train_data_Y = Load_Data(data, TrimDict)

c = 0
cl1_hit = 0
cl2_hit = 0
for row in train_data_X:

    if train_data_Y[c] == 1:
        if cl1_hit == 0:
            plt.plot(row[0], row[1],  'o', color='#009cffff', markersize=mks, label="Class 1")
            cl1_hit = 1
        else:
            plt.plot(row[0], row[1],  'o', color='#009cffff', markersize=mks)
    else:
        if cl2_hit == 0:
            plt.plot(row[0], row[1],  '*', alpha=0.8, color='#ff8800ff', markersize=mks_star, label="Class 2")
            cl2_hit = 1
        else:
            plt.plot(row[0], row[1],  '*', alpha=0.8, color='#ff8800ff', markersize=mks_star)
    c = c + 1
plt.legend(markerscale=1.5, fancybox=True)  # , facecolor='k'

plt.axis('equal')

if train_ref != 'na':
    fig.text(0.02, 0.94, train_ref, fontsize=14, fontweight='bold')

train_data_X, train_data_Y = Load_Data(data, TrimDict, noise=5, noise_type='per')
#plt.scatter(train_data_X[:,0], train_data_X[:,1], c=train_data_Y, s=2)

#matplotlib.rcParams['axes.prop_cycle'] = cycler('color', ['r', 'k'])

basic_cols = ['#009cff', '#6d55ff', '#ffffff', '#ff6d55','#ff8800']  # pastal orange/red/white/purle/blue
basic_cols = ['#6d55ff', '#ff6d55']  # pastal orange/red/white/purle/blue
my_cmap = LinearSegmentedColormap.from_list('mycmap', basic_cols)

plt.scatter(train_data_X[:,0], train_data_X[:,1],  s=3, c=train_data_Y, cmap=my_cmap)

plt.show()
exit()

# # show & save plots
fig_path = "graphs/%s_%s.png" % (data, set)
fig.savefig(fig_path, dpi=600)

fig_path = "graphs/%s_%s.svg" % (data, set)
fig.savefig(fig_path, dpi=600)

plt.close(fig)



# # # # plot responce data # # # #
fig = plt.figure(2)

plt.xlabel('$a_1$')
plt.ylabel('$a_2$')

# plot test data too
data = 'veri'
train_data_X, train_data_Y = Load_Data(data, TrimDict)
c = 0
cl1_hit = 0
cl2_hit = 0
for row in train_data_X:

    if train_data_Y[c] == 1:
        if cl1_hit == 0:
            plt.plot(row[0], row[1],  'o', color='#009cffff', markersize=mks, label="Class 1")
            cl1_hit = 1
        else:
            plt.plot(row[0], row[1],  'o', color='#009cffff', markersize=mks)
    else:
        if cl2_hit == 0:
            plt.plot(row[0], row[1],  '*', alpha=0.8, color='#ff8800ff', markersize=mks_star, label="Class 2")
            cl2_hit = 1
        else:
            plt.plot(row[0], row[1],  '*', alpha=0.8, color='#ff8800ff', markersize=mks_star)
    c = c + 1
plt.legend(markerscale=1.5, fancybox=True)  # , facecolor='k'

plt.axis('equal')

if veri_ref != 'na':
    fig.text(0.02, 0.94, veri_ref, fontsize=14, fontweight='bold')

# # show & save plots
fig_path = "graphs/%s_%s.png" % (data, set)
fig.savefig(fig_path, dpi=600)

fig_path = "graphs/%s_%s.svg" % (data, set)
fig.savefig(fig_path, dpi=600)

plt.close(fig)








#

# fin
