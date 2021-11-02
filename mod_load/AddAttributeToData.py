import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import os
import matplotlib
import h5py

from mod_settings.Set_Load import LoadSettings

"""
A function which takes a list of Network responces, each corresponding to a
specific instance, and adds a column to the data (before the final binary
column).

This is so the EiM processor can add an atribute which represents a confidence
or closeness to the boundary between classes which was evolved.
"""


def AddAttribute(prm, data_type, responceY, best_fit, sys, rep, train_data_X, train_data_Y, add_to_file_name=''):

    # type it either 'train', or 'veri'

    matplotlib .rcParams['font.family'] = 'Arial'  # 'serif'
    matplotlib .rcParams['font.size'] = 9  # tixks and title
    matplotlib .rcParams['figure.titlesize'] = 'medium'
    matplotlib .rcParams['axes.labelsize'] = 11  # axis labels
    matplotlib .rcParams['axes.linewidth'] = 1
    matplotlib .rcParams["legend.labelspacing"] = 0.25
    matplotlib.rc('pdf', fonttype=42)  # embeds the font, so can import to inkscape
    matplotlib .rcParams["figure.autolayout"] = True  # False

    ParamDict = prm['DE']
    NetworkDict = prm['network']


    # # sort the weigts into a list
    if len(responceY.shape) == 2:
        if responceY.shape[1] == 1:
            responceY = responceY[:,0]
        else:
            print("Warning: could not generate extra attribure. Responce is the wrong shape: %s" % str(responceY.shape))
            return


    # format input data
    for i in range(len(train_data_X[0,:])):
        if i == 0:
            att = 'Attribute%d' % (i+1)
            df1 = pd.DataFrame({att: train_data_X[:,i]})
        else:
            att = 'Attribute%d' % (i+1)
            df11 = pd.DataFrame({att: train_data_X[:,i]})
            df1 = pd.concat([df1, df11], axis=1)

    #print("df1:\n", df1.head())

    if ParamDict['IntpScheme'] != 'band' and ParamDict['IntpScheme'] != 'HOW':
        new_data_label = 'Class'
    else:
        new_data_label = 'Responce'
    df2 = pd.DataFrame({'NewAttribute': responceY,
                      'class': train_data_Y})


    #print("df2:\n", df2.head())

    data_all = pd.concat([df1, df2], axis=1)

    #print("df all:\n", data_all.head())


    # make save dir
    dir = save_loc = "%s/NewData" % prm['SaveDir']
    if not os.path.exists(dir):
        os.makedirs(dir)


    # # save
    if ParamDict['SaveExtraAttribute'] == 1:
        save_loc = "%s/DataWithWeigthAttribute.h5" % (dir)
        data_all.to_hdf(save_loc, key="sys%d_rep%d/%s" % (sys, rep, data_type), mode='a')


    # # plot if it is toggled on
    if ParamDict['PlotExtraAttribute'] == 1:



        colr_scale_lims = 0.2  # the +/- values that below which it starts to fade to black (i.e a zero responce)



        num_attr = len(train_data_X[0, :])

        # ###################################################
        # # Plot original data with colour weights # #
        # ###################################################
        """
        rows, cols = num_attr-1, num_attr-1
        fig, ax = plt.subplots(rows, cols, sharex='col', sharey='row', squeeze=False)


        if ParamDict['training_data'] == 'MMDS':
            add_att = 2
        else:
            add_att = 1

        #print("PLOTTING 4d")
        for i in range(rows+1):
            #print("i", i)
            for j in range(i+1, cols+1):

                # plot this sub graph
                for k in range(len(data_all)):
                    current_class = data_all.loc[k, 'class']
                    current_BW = data_all.loc[k, 'NewAttribute']
                    x_att = 'Attribute%d' % (i+1)
                    y_att = 'Attribute%d' % (j+1)
                    current_x = data_all.loc[k, x_att]
                    current_y = data_all.loc[k, y_att]

                    if ParamDict['IntpScheme'] != 'band' and ParamDict['IntpScheme'] != 'HOW' and ParamDict['IntpScheme'] != 'clustering' and ParamDict['IntpScheme'] != 'RidgeHOW':
                        if current_BW >= colr_scale_lims:
                            colr = (1, 0, 0)
                        elif current_BW <= -colr_scale_lims:
                            colr = (0, 0, 1)
                        else:
                            fac = current_BW/colr_scale_lims
                            if fac >= 0:
                                colr = (abs(fac), 0, 0)
                            elif fac < 0:
                                colr = (0, 0, abs(fac))
                    else:
                        if current_BW == 2:
                            colr = (1, 0, 0)
                        elif current_BW == 1:
                            colr = (0, 0, 1)
                        elif current_BW == 3:
                            colr = (0, 1, 0)
                        else:
                            print("\n>>Intp Scheme: ", ParamDict['IntpScheme'])
                            print(">> current_BW = ", current_BW, "\n")


                    if current_class == 2:
                        cl2, = ax[j-1, i].plot(current_x, current_y,  '*', color=colr, markersize=5, alpha=0.92, markeredgewidth=0.5, markeredgecolor=(1, 0, 0, 1))
                    elif current_class == 1:
                        cl1, = ax[j-1, i].plot(current_x, current_y,  'o', color=colr, markersize=3.5, alpha=0.92, markeredgewidth=0.5, markeredgecolor=(0, 0, 1, 1))
                    elif current_class == 3:
                        cl3, = ax[j-1, i].plot(current_x, current_y,  '^', color=colr, markersize=3.5, alpha=0.92, markeredgewidth=0.5, markeredgecolor=(0, 1, 0, 1))



                if j == cols:
                    x_lab = "Attr %d" % (i+add_att)
                    ax[j-1, i].set_xlabel(x_lab, fontsize=12)

                if i == 0:
                    y_lab = "Attr %d" % (j+add_att)
                    ax[j-1, i].set_ylabel(y_lab, fontsize=12)


        for in_i in range(num_attr-1):
            j_range = np.arange(in_i)
            for in_j in j_range:
                fig.delaxes(ax[in_j][in_i])

        #plt.legend([cl1, cl2, cl3], ["Class 1", "Class 2", "Class 3"])

        fig.subplots_adjust(hspace=0.1, wspace=0.1)
        #fig.set_size_inches(15,10)
        fig.set_size_inches(6.4, 4.8)
        the_ti = 'Predictive attributs, output represented as a colour (r/b >= +/-%f)\n(%s data) Fitness: %.3f' % (colr_scale_lims, data_type, best_fit)
        fig.suptitle(the_ti)
        fig1_path = "%s/%d_Rep%d_%s_DataWithWeigthAttribute%s.png" % (dir, sys, rep, data_type, add_to_file_name)
        fig.savefig(fig1_path, dpi=150)

        # """

        #

        #

        #

        #

        # # #########################################################
        # # Plot original data with weights as an extra attribute # #
        # ###########################################################
        rows, cols = num_attr, num_attr
        fig, ax = plt.subplots(rows, cols, sharex='col', sharey='row')

        if ParamDict['training_data'] == 'MMDS':
            add_att = 2
            new_att = "Attr %d" % (num_attr+2)
        else:
            add_att = 1
            new_att = "Attr %d" % (num_attr+1)

        new_attr = 'Attribute%d' % (num_attr+1)


        for i in range(rows+1):
            #print("i", i)
            for j in range(i+1, cols+1):

                x_att = 'Attribute%d' % (i+1)
                y_att = 'Attribute%d' % (j+1)

                if x_att == new_attr:
                    x_att = 'NewAttribute'
                elif y_att == new_attr:
                    y_att = 'NewAttribute'

                # plot class 2
                dat_class2 = data_all[data_all['class'] == 2]
                ax[j-1, i].plot(dat_class2[x_att], dat_class2[y_att],  '*', color='r', markersize=5.25, alpha=0.3, markeredgewidth=0)

                # plot class 1
                dat_class1 = data_all[data_all['class'] == 1]
                ax[j-1, i].plot(dat_class1[x_att], dat_class1[y_att],  'o', color='b', markersize=3.5, alpha=0.3, markeredgewidth=0)

                dat_class3 = data_all[data_all['class'] == 3]
                ax[j-1, i].plot(dat_class3[x_att], dat_class3[y_att],  '^', color='g', markersize=3.5, alpha=0.3, markeredgewidth=0)

                if j == cols:
                    x_lab = "Attr %d" % (i+add_att)
                    if x_lab == new_att:
                        x_lab = "New Attr"
                    ax[j-1, i].set_xlabel(x_lab, fontsize=12)

                if i == 0:
                    y_lab = "Attr %d" % (j+add_att)
                    if y_lab == new_att:
                        y_lab = "New Attr"
                    ax[j-1, i].set_ylabel(y_lab, fontsize=12)

        for in_i in range(num_attr):
            j_range = np.arange(in_i)
            for in_j in j_range:
                fig.delaxes(ax[in_j][in_i])

        fig.subplots_adjust(hspace=0.1, wspace=0.1)
        #fig.set_size_inches(15,10)
        fig.set_size_inches(6.4, 4.8)
        the_ti = 'Original Data plotted with generated output\n(%s data) Fitness: %.3f' % (data_type, best_fit)
        fig.suptitle(the_ti)
        fig1_path = "%s/%d_Rep%d_%s_DataWithWeigthAttribute_alt%s.png" % (dir, sys, rep, data_type, add_to_file_name)
        fig.savefig(fig1_path, dpi=150)

    #

    plt.close('all')

    return

#

#

#

#

#

# fin
