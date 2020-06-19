import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import os

from Module_LoadData.LoadData import Load_Data
from Module_Settings.Set_Load import LoadSettings

"""
A function which takes a list of Network responces, each corresponding to a
specific instance, and adds a column to the data (before the final binary
column).

This is so the EiM processor can add an atribute which represents a confidence
or closeness to the boundary between classes which was evolved.
"""


def AddAttribute(type, responceY):

    # type it either 'train', or 'veri'

    ParamDict = LoadSettings()

    if type == 'train':
        if ParamDict['TestVerify'] == 0:
            data_type = 'all'
        elif ParamDict['TestVerify'] == 1:
            data_type = 'training'
    elif type == 'veri':
        data_type = 'veri'

    # # sort the weigts into a list
    BW_list = []
    for ele in responceY:
        BW_list.append(ele[0])


    # # # # # # # # # # # #
    # Load the training data
    train_data_X, train_data_Y = Load_Data(type, ParamDict['num_input'],
                                           ParamDict['num_output_readings'],
                                           ParamDict['training_data'],
                                           ParamDict['TestVerify'],
                                           ParamDict['UseCustom_NewAttributeData'])

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

    df2 = pd.DataFrame({'NewAttribute': BW_list,
                      'class': train_data_Y})


    #print("df2:\n", df2.head())

    data_all = pd.concat([df1, df2], axis=1)

    #print("df all:\n", data_all.head())


    # make save dir
    dir = save_loc = "%s/NewData" % ParamDict['SaveDir']
    if not os.path.exists(dir):
        os.makedirs(dir)


    # save
    save_loc = "%s/%d_Rep%d_DataWithWeigthAttribute.h5" % (dir, ParamDict['circuit_loop'], ParamDict['repetition_loop'])
    data_all.to_hdf(save_loc, key=data_type, mode='a')


    # plot if it is toggled on
    if ParamDict['PlotExtraAttribute'] == 1:



        colr_scale_lims = 0.2  # the +/- values that below which it starts to fade to black (i.e a zero responce)

        # *********************************************************************
        # 2d data plot
        # ********************************************************************
        if len(train_data_X[0, :]) == 2:
            #print("PLOTTING 2d")

            fig = plt.figure()  # produce figure

            temp_c1 = []
            temp_c2 = []

            for i in range(len(data_all)):
                current_class = data_all.loc[i, 'class']
                current_BW = data_all.loc[i, 'NewAttribute']
                current_x1 = data_all.loc[i, 'Attribute1']
                current_x2 = data_all.loc[i, 'Attribute2']

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

                if current_class == 2:
                    plt.plot(current_x1, current_x2,  '*', color=colr, markersize=5, alpha=0.9, markeredgewidth=0.3, markeredgecolor=(1, 0, 0, 1))
                    temp_c1.append([current_x1, current_x2, current_BW])
                elif current_class == 1:
                    plt.plot(current_x1, current_x2,  'o', color=colr, markersize=3.5, alpha=0.9, markeredgewidth=0.3, markeredgecolor=(0, 0, 1, 1))
                    temp_c2.append([current_x1, current_x2, current_BW])

            the_ti = "Weighted data (black=0, r/b >= +/-%f)" % (colr_scale_lims)
            plt.title(the_ti)
            plt.xlabel('x1')
            plt.ylabel('x2')

            fig1_path = "%s/%d_Rep%d_%s_DataWithWeigthAttribute.pdf" % (dir, ParamDict['circuit_loop'], ParamDict['repetition_loop'], type)
            fig.savefig(fig1_path)

            # # Plot original data with weights as an extra attribute
            rows, cols = 2, 2
            fig, ax = plt.subplots(rows, cols, sharex='col', sharey='row')

            add_att = 1
            new_att = "Attr 3"

            #print("PLOTTING 2d alt")
            for i in range(rows+1):
                #print("i", i)
                for j in range(i+1, cols+1):

                    x_att = 'Attribute%d' % (i+1)
                    y_att = 'Attribute%d' % (j+1)

                    if x_att == 'Attribute3':
                        x_att = 'NewAttribute'
                    elif y_att == 'Attribute3':
                        y_att = 'NewAttribute'

                    # plot class 2
                    dat_class2 = data_all[data_all['class'] == 2]
                    ax[j-1, i].plot(dat_class2[x_att], dat_class2[y_att],  '*', color='r', markersize=5.25, alpha=0.3, markeredgewidth=0)

                    # plot class 1
                    dat_class1 = data_all[data_all['class'] == 1]
                    ax[j-1, i].plot(dat_class1[x_att], dat_class1[y_att],  'o', color='b', markersize=3.5, alpha=0.3, markeredgewidth=0)

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

            fig.delaxes(ax[0][1])

            fig.subplots_adjust(hspace=0.1, wspace=0.1)
            fig.set_size_inches(15,10)
            the_ti = 'Predictive attributs with border weights (red/star=class2, blue/o=class1)'
            fig.suptitle(the_ti)
            fig1_path = "%s/%d_Rep%d_%s_DataWithWeigthAttribute_alt.pdf" % (dir, ParamDict['circuit_loop'], ParamDict['repetition_loop'], type)
            fig.savefig(fig1_path)

            # # # # # ## # ## # ## # ## ## # ##
            # # Make a 3d plot drom the data

            ax = plt.axes(projection='3d')

            temp_c1 = np.asarray(temp_c1)
            temp_c2 = np.asarray(temp_c2)

            ax.scatter3D(temp_c1[:,0], temp_c1[:,1], abs(temp_c1[:,2]), marker='o', color='b')  # depthshade=False
            ax.scatter3D(temp_c2[:,0], temp_c2[:,1], abs(temp_c2[:,2]), marker='*', color='r')  # depthshade=False2

            ax.set_xlabel('Attr 1')
            ax.set_ylabel('Attr 2')
            ax.set_zlabel('Magnitude of New Attr (i.e Y)')

            fig.subplots_adjust(hspace=0.1, wspace=0.1)
            fig.set_size_inches(15,10)
            the_ti = '3d plot of original 2 Attributes, and new Y attribute'
            fig.suptitle(the_ti)
            fig1_path = "%s/%d_Rep%d_%s_DataWithWeigthAttribute_alt_3d.pdf" % (dir, ParamDict['circuit_loop'], ParamDict['repetition_loop'], type)
            fig.savefig(fig1_path)

        # ********************************************************************
        # 3 attricute data plot
        # ********************************************************************
        elif len(train_data_X[0, :]) == 3:

            # # Plot original data with colour weights
            rows, cols = 2, 2
            fig, ax = plt.subplots(rows, cols, sharex='col', sharey='row')
            add_att = 1

            #print("PLOTTING 3d")
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

                        if current_class == 2:
                            ax[j-1, i].plot(current_x, current_y,  '*', color=colr, markersize=5.25, alpha=0.3, markeredgewidth=0)

                        elif current_class == 1:
                            ax[j-1, i].plot(current_x, current_y,  'o', color=colr, markersize=3.75, alpha=0.3, markeredgewidth=0)


                    if j == cols:
                        x_lab = "Attr %d" % (i+add_att)
                        ax[j-1, i].set_xlabel(x_lab, fontsize=12)

                    if i == 0:
                        y_lab = "Attr %d" % (j+add_att)
                        ax[j-1, i].set_ylabel(y_lab, fontsize=12)

            fig.delaxes(ax[0][1])
            #fig.delaxes(ax[0][2])
            #fig.delaxes(ax[1][2])

            fig.subplots_adjust(hspace=0.1, wspace=0.1)
            fig.set_size_inches(15,10)
            the_ti = 'Predictive attributs With bourder weights (black=0 or boundary, r/b >= +/-%f)' % (colr_scale_lims)
            fig.suptitle(the_ti)
            fig1_path = "%s/%d_Rep%d_%s_DataWithWeigthAttribute.pdf" % (dir, ParamDict['circuit_loop'], ParamDict['repetition_loop'], type)
            fig.savefig(fig1_path)

            # # # # # ## # ## # ## # ## ## # ##
            # # Plot original data with weights as an extra attribute
            rows, cols = 3, 3
            fig, ax = plt.subplots(rows, cols, sharex='col', sharey='row')

            add_att = 1
            new_att = "Attr 4"

            #print("PLOTTING 3d alt")
            for i in range(rows+1):
                #print("i", i)
                for j in range(i+1, cols+1):

                    x_att = 'Attribute%d' % (i+1)
                    y_att = 'Attribute%d' % (j+1)

                    if x_att == 'Attribute4':
                        x_att = 'NewAttribute'
                    elif y_att == 'Attribute4':
                        y_att = 'NewAttribute'

                    # plot class 2
                    dat_class2 = data_all[data_all['class'] == 2]
                    ax[j-1, i].plot(dat_class2[x_att], dat_class2[y_att],  '*', color='r', markersize=5.25, alpha=0.3, markeredgewidth=0)

                    # plot class 1
                    dat_class1 = data_all[data_all['class'] == 1]
                    ax[j-1, i].plot(dat_class1[x_att], dat_class1[y_att],  'o', color='b', markersize=3.5, alpha=0.3, markeredgewidth=0)

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

            fig.delaxes(ax[0][1])
            fig.delaxes(ax[0][2])
            fig.delaxes(ax[1][2])
            #fig.delaxes(ax[0][3])
            #fig.delaxes(ax[1][3])
            #fig.delaxes(ax[2][3])

            fig.subplots_adjust(hspace=0.1, wspace=0.1)
            fig.set_size_inches(15,10)
            the_ti = 'Predictive attributs with border weights (red/star=class2, blue/o=class1)'
            fig.suptitle(the_ti)
            fig1_path = "%s/%d_Rep%d_%s_DataWithWeigthAttribute_alt.pdf" % (dir, ParamDict['circuit_loop'], ParamDict['repetition_loop'], type)
            fig.savefig(fig1_path)



        # ********************************************************************
        # 4d or more data plot
        # ********************************************************************
        elif len(train_data_X[0, :]) >= 4:

            num_attr = len(train_data_X[0, :])

            # # Plot original data with colour weights
            rows, cols = num_attr-1, num_attr-1
            fig, ax = plt.subplots(rows, cols, sharex='col', sharey='row')

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

                        if current_class == 2:
                            ax[j-1, i].plot(current_x, current_y,  '*', color=colr, markersize=5.25, alpha=0.3, markeredgewidth=0)
                        elif current_class == 1:
                            ax[j-1, i].plot(current_x, current_y,  'o', color=colr, markersize=3.75, alpha=0.3, markeredgewidth=0)



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



            fig.subplots_adjust(hspace=0.1, wspace=0.1)
            fig.set_size_inches(15,10)
            the_ti = 'Predictive attributs With bourder weights as colours (black=0 or boundary, r/b >= +/-%f)' % (colr_scale_lims)
            fig.suptitle(the_ti)
            fig1_path = "%s/%d_Rep%d_%s_DataWithWeigthAttribute.pdf" % (dir, ParamDict['circuit_loop'], ParamDict['repetition_loop'], type)
            fig.savefig(fig1_path)

            # # # ## # # # ## ## # #
            # # Plot original data with weights as an extra attribute
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
            fig.set_size_inches(15,10)
            the_ti = 'Predictive attributs with border weights (red/star=class2, blue/o=class1)'
            fig.suptitle(the_ti)
            fig1_path = "%s/%d_Rep%d_%s_DataWithWeigthAttribute_alt.pdf" % (dir, ParamDict['circuit_loop'], ParamDict['repetition_loop'], type)
            fig.savefig(fig1_path)

    #

    plt.close('all')

    return

#

#

#

#

#

# fin
