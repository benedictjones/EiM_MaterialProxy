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

from FetchDataObj import FetchDataObj


matplotlib .rcParams['font.family'] = 'Arial'  # 'serif'
matplotlib .rcParams['font.size'] = 8  # tixks and title
matplotlib .rcParams['figure.titlesize'] = 'medium'
matplotlib .rcParams['axes.labelsize'] = 10  # axis labels
matplotlib .rcParams['axes.linewidth'] = 1  # box edge
#matplotlib .rcParams['mathtext.fontset'] = 'Arial'  # 'cm'
matplotlib.rc('pdf', fonttype=42)  # embeds the font, so can import to inkscape
matplotlib .rcParams["legend.labelspacing"] = 0.25
matplotlib .rcParams["legend.fontsize"] = 8

#matplotlib .rcParams['lines.linewidth'] = 0.85
matplotlib .rcParams['lines.markersize'] = 3.5
matplotlib .rcParams['lines.markeredgewidth'] = 0.5

matplotlib .rcParams["figure.figsize"] = [2.5,2.2]
matplotlib .rcParams["figure.autolayout"] = True
# ********************************************************************

train_ref = 'na'
veri_ref = 'na'

sets = ['d2DDS', '2DDS', 'XOR', 'con2DDS', 'hm2DDS', 'c2DDS']
sets = ['d2DDS', 'c2DDS', 'flipped_d2DDS']
sets = ['flipped_d2DDS']

for set in sets:

    TrimDict = {}
    TrimDict['DE'] = {}
    TrimDict['DE']['TestVerify'] = 1
    TrimDict['DE']['UseCustom_NewAttributeData'] = 0
    TrimDict['DE']['training_data'] = set
    TrimDict['DE']['FitScheme'] = 'error'
    TrimDict['DE']['batch_size'] = 1
    TrimDict['DE']['batch_scheme'] = 'none'
    TrimDict['DE']['batch_window_size'] = 'na'
    TrimDict['DE']['data_oversample'] = 0
    TrimDict['DE']['data_weighting'] = 0

    TrimDict['spice'] = {}
    TrimDict['spice']['Vmin'] = -5
    TrimDict['spice']['Vmax'] = 5


    lobj = FetchDataObj(TrimDict)

    """lobj = FetchDataObj(TrimDict, batch_size=0)
    x, y = lobj.fetch_data('train')
    print(x)
    exit()

    # Test batching
    lobj = FetchDataObj(TrimDict, batch_size=40)

    for i in range(25):
        x, y = lobj.fetch_data('train', iterate_batch=1)
        #print(lobj.b_epoch, lobj.b_idx, lobj.n_comp, "\n", x, y)
        print(lobj.b_epoch, lobj.b_idx, lobj.n_comp, "batch size:", len(x), "\n")

        # lobj.__init__(TrimDict)  # re-set!!
    exit()"""

    """if set == '2DDS' or set == 'flipped_2DDS':
        the_axis = [0.5, 3.5, 0.5, 3.5]  # 2DDS
    elif set == 'con2DDS':
        the_axis = [-4, 4, -4, 4]  # con2DDS"""


    mks = 3.5
    mks_star = mks + 1.5

    data = 'train'
    train_data_X, train_data_Y = lobj.fetch_data(data)

    unique, counts = np.unique(train_data_Y, return_counts=True)
    d = dict(zip(unique, counts))
    print("Training data counts", d)

    # # # # plot responce data # # # #
    fig = plt.figure()

    plt.xlabel('$a_1$')
    plt.ylabel('$a_2$')


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

    train_data_X, train_data_Y = lobj.fetch_data(data, noise=5, noise_type='per')
    #plt.scatter(train_data_X[:,0], train_data_X[:,1], c=train_data_Y, s=2)

    #matplotlib.rcParams['axes.prop_cycle'] = cycler('color', ['r', 'k'])

    basic_cols = ['#009cff', '#6d55ff', '#ffffff', '#ff6d55','#ff8800']  # pastal orange/red/white/purle/blue
    basic_cols = ['#6d55ff', '#ff6d55']  # pastal orange/red/white/purle/blue
    my_cmap = LinearSegmentedColormap.from_list('mycmap', basic_cols)

    plt.scatter(train_data_X[:,0], train_data_X[:,1],  s=3, c=train_data_Y, cmap=my_cmap)

    """plt.show()
    exit()"""

    """# # show & save plots
    fig_path = "graphs/%s_%s.png" % (data, set)
    fig.savefig(fig_path, dpi=600)

    fig_path = "graphs/%s_%s.svg" % (data, set)
    fig.savefig(fig_path, dpi=600)"""

    #plt.close(fig)
    plt.axis('equal')


    # # # # plot responce data # # # #
    fig = plt.figure()

    plt.xlabel('$a_1$')
    plt.ylabel('$a_2$')

    # plot test data too
    data = 'test'
    train_data_X, train_data_Y = lobj.fetch_data(data)
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

    """# # show & save plots
    fig_path = "graphs/%s_%s.png" % (data, set)
    fig.savefig(fig_path, dpi=600)

    fig_path = "graphs/%s_%s.svg" % (data, set)
    fig.savefig(fig_path, dpi=600)

    plt.close(fig)"""


    unique, counts = np.unique(train_data_Y, return_counts=True)
    d = dict(zip(unique, counts))
    print("Test data counts", d)








    # # # # plot responce data # # # #
    fig = plt.figure()

    plt.xlabel('$a_1$')
    plt.ylabel('$a_2$')

    # plot test data too
    data = 'validation'
    train_data_X, train_data_Y = lobj.get_data(data, iterate=0)
    plt.title('The %s (%d instances)' % (data, len(train_data_X)))
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

    #""" # # show & save plots
    fig_path = "mod_load/graphs/%s_%s.png" % (data, set)
    fig.savefig(fig_path, dpi=600)

    #fig_path = "graphs/%s_%s.svg" % (data, set)
    #fig.savefig(fig_path, dpi=600)

    plt.close(fig)
    #"""


    unique, counts = np.unique(train_data_Y, return_counts=True)
    d = dict(zip(unique, counts))
    print("Validation data counts", d)







    # # # # plot responce data # # # #
    fig = plt.figure()

    plt.xlabel('$a_1$')
    plt.ylabel('$a_2$')

    # plot test data too
    data = 'test'
    train_data_X, train_data_Y = lobj.get_data(data, iterate=0)
    plt.title('The %s (%d instances)' % (data, len(train_data_X)))
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



    unique, counts = np.unique(train_data_Y, return_counts=True)
    d = dict(zip(unique, counts))
    print("Test data counts", d)

    #""" # # show & save plots
    fig_path = "mod_load/graphs/%s_%s.png" % (data, set)
    fig.savefig(fig_path, dpi=600)

    #fig_path = "graphs/%s_%s.svg" % (data, set)
    #fig.savefig(fig_path, dpi=600)

    plt.close(fig)
    #"""





    # # # # plot responce data # # # #
    for noise in [5, 10, 20]:
        fig = plt.figure()

        plt.xlabel('$a_1$')
        plt.ylabel('$a_2$')

        # plot test data too
        data = 'test'
        train_data_X, train_data_Y = lobj.fetch_data(data, iterate=0, noise=noise, noise_type='per')
        plt.title('The %s (%d instances), Noise=%f' % (data, len(train_data_X), noise))
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



        unique, counts = np.unique(train_data_Y, return_counts=True)
        d = dict(zip(unique, counts))
        print("Test data counts", d)

        #""" # # show & save plots
        fig_path = "mod_load/graphs/%s_%s_Noise%d.png" % (data, set, noise)
        fig.savefig(fig_path, dpi=600)

        #fig_path = "graphs/%s_%s.svg" % (data, set)
        #fig.savefig(fig_path, dpi=600)

        plt.close(fig)
        #"""













    # # # # plot responce data # # # #
    fig = plt.figure()

    plt.xlabel('$a_1$')
    plt.ylabel('$a_2$')

    # plot test data too
    data = 'train'
    train_data_X, train_data_Y = lobj.get_data(data, iterate=0)
    plt.title('The %s (%d instances)' % (data, len(train_data_X)))
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

    unique, counts = np.unique(train_data_Y, return_counts=True)
    d = dict(zip(unique, counts))
    print("Training data counts", d)

    #""" # # show & save plots
    fig_path = "mod_load/graphs/%s_%s.png" % (data, set)
    fig.savefig(fig_path, dpi=600)

    #fig_path = "graphs/%s_%s.svg" % (data, set)
    #fig.savefig(fig_path, dpi=600)

    plt.close(fig)
    #"""







    # # # # plot responce data # # # #
    fig = plt.figure()

    plt.xlabel('$a_1$')
    plt.ylabel('$a_2$')

    # plot test data too
    data = 'all'
    data_X, data_Y = lobj.get_data(data, iterate=0)
    # plt.title('The %s (%d instances)' % (data, len(train_data_X)))
    c = 0
    cl1_hit = 0
    cl2_hit = 0
    markers = ["o", "*", "x", "v", "+", "." , "^" , "<", ">"]
    colours = ['#009cffff','#ff8800ff','#9cff00ff','c','m', 'y', 'k']
    sizes = [2.5, 4.5, 2.5, 2.5, 3.5, 3.5, 3.5]
    class_values = np.unique(data_Y)
    class_values.sort()
    # Group Classes
    class_data = []
    for cla in class_values:
        class_group = []
        for instance, yclass in enumerate(data_Y):
            if yclass == cla:
                class_group.append([data_X[instance,0], data_X[instance,1]])
        class_group = np.asarray(class_group)
        class_data.append(class_group)

    # Plot the grouped classes
    for idx, cla in enumerate(class_values):
        inst_data = class_data[idx]
        plt.plot(inst_data[:,0], inst_data[:,1], color=colours[idx],
                 marker=markers[idx], markersize=sizes[idx], ls='',
                 label="Class %d" % (idx+1),
                 alpha=0.8, markeredgewidth=0.2, markeredgecolor=(1, 1, 1, 1))

    plt.legend(markerscale=1.2, fancybox=True)  # , facecolor='k'

    plt.axis('equal')

    plt.xlim([np.min(data_X[:,0])-1, np.max(data_X[:,0])+1])
    plt.ylim([np.min(data_X[:,1])-1, np.max(data_X[:,1])+1])

    if veri_ref != 'na':
        fig.text(0.02, 0.94, veri_ref, fontsize=14, fontweight='bold')

    unique, counts = np.unique(train_data_Y, return_counts=True)
    d = dict(zip(unique, counts))
    print("Training data counts", d)

    #""" # # show & save plots
    fig_path = "mod_load/graphs/%s_%s.png" % (data, set)
    fig.savefig(fig_path, dpi=600)

    #fig_path = "graphs/%s_%s.svg" % (data, set)
    #fig.savefig(fig_path, dpi=600)

    plt.close(fig)
    #"""


    #plt.show()



#

# fin
