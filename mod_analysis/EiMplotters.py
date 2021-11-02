import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FormatStrFormatter

from mpl_toolkits import mplot3d

from mod_methods.RoundDown import round_decimals_down
import numpy as np
import sklearn.cluster as cluster
import sklearn.metrics as met

from sklearn.decomposition import PCA
from sklearn import preprocessing

import matplotlib.animation as animation
import time
# matplotlib.use('Agg')

from mod_methods.MovingAverage import moving_average, moving_convolution

from mod_material.spice.NodeToOps_LoadRun import LoadRun_DC_InNodeToOps
from mod_material.spice.TransConductance_LoadRun import LoadRun_DC_TransConductance

######################################################
# # # EiM Plotter # # #
# Contains various functions to plot EiM  results
######################################################


#

#

#
def FitEvo(x, prm, syst, rep, training_fits, TestFit, Global_bFits, BatchPerEpoch,
                raw_vali_bFits, batch_n_comp_list, epoch_n_comp_list, num_batches):
    """
    Plots A Evolution of the accuracy (i.e. data error)
    Can select the x axis to be Ncomps or Epochs.
    """

    if BatchPerEpoch == 0:
        batch_weight = 1
    else:
        batch_weight = 1/BatchPerEpoch

    if x == 'Epoch':
        new_stop = batch_weight*(len(training_fits)-1)
        nspaces = len(training_fits)
        train_x_vals = np.linspace(0, new_stop, nspaces)

        vali_x_vals = np.arange(1, len(raw_vali_bFits)+1, 1)
    elif x == 'Ncomps':
        train_x_vals = batch_n_comp_list
        vali_x_vals = epoch_n_comp_list

    # # Don't Plot if no epochs where carried out
    if prm['DE']['epochs'] == 0:
        print("Won't plot FitEvo, as no epochs where carried out")
        return

    if prm['DE']['save_fitvit'] == 1:
        lines = []

        fig_evole = plt.figure(tight_layout=False)
        axes_evole = fig_evole.add_axes([0.125, 0.125, 0.75, 0.75])

        # Plot train pop best fit
        color = 'tab:blue'
        ln1 = axes_evole.plot(train_x_vals, training_fits, '--', color=color, label='Train $p_{best}$', alpha=0.6)  # NOTE: This includes the best fitness of original population
        lines = ln1

        # Add trend line to train pop best fit
        if prm['DE']['batch_size'] != 0 and prm['DE']['batch_size'] != 1:
            av = moving_convolution(training_fits, N=num_batches)
            ln2 = axes_evole.plot(train_x_vals, av, '-', color=color, label='Training trend (convolution)', alpha=0.85)
            lines = lines + ln2

        if x == 'Epoch':
            if prm['DE']['batch_size'] == 0:
                axes_evole.set_xlabel('Generation')
            elif prm['DE']['batch_size'] == 1:
                axes_evole.set_xlabel('Epoch (Batch Size = 1 * TrainingData))')
            elif prm['DE']['batch_size'] < 1:
                axes_evole.set_xlabel('Epoch (Batch size = %.3f * TrainingData)' % (prm['DE']['batch_size']))
            else:
                axes_evole.set_xlabel('Epoch (Batch size = %d )' % (prm['DE']['batch_size']))
        elif x == 'Ncomps':
            if prm['DE']['batch_size'] == 0:
                axes_evole.set_xlabel('Ncomps')
            elif prm['DE']['batch_size'] == 1:
                axes_evole.set_xlabel('Ncomps (Batch Size = 1 * TrainingData))')
            elif prm['DE']['batch_size'] < 1:
                axes_evole.set_xlabel('Ncomps (Batch size = %.3f * TrainingData)' % (prm['DE']['batch_size']))
            else:
                axes_evole.set_xlabel('Ncomps (Batch size = %d )' % (prm['DE']['batch_size']))

        axes_evole.set_ylabel('Fitness (%s)' % (prm['DE']['FitScheme']))

        #axes_evole.set_xlim([0, train_x_vals[-1]+1])

        # set tital
        new_title = 'Fitness (loss) Evolution\n Test Error = %.4f' % (TestFit)
        axes_evole.set_title(new_title)

        # # Plot Validation accuracy
        if prm['DE']['batch_size'] != 0:

            if prm['DE']['Gbest_fit'] != 'raw' and prm['DE']['FitScheme'] != prm['DE']['Gbest_fit']:

                axes_evole.tick_params(axis='y', labelcolor=color)
                axes_evole.set_ylabel('Training Fitness (%s)' % (prm['DE']['FitScheme']), color=color)

                axt = axes_evole.twinx()
                color = 'tab:red'
                axt.set_ylabel('Validation Fitness (%s)' % (prm['DE']['Gbest_fit']), color=color)  # we already handled the x-label with ax1
                axt.tick_params(axis='y', labelcolor=color)
            else:
                axes_evole.set_ylabel('Fitness (%s)' % (prm['DE']['FitScheme']))
                axt = axes_evole
                color = 'tab:red'

            ln = axt.plot(vali_x_vals, Global_bFits, '-*', color=color, label='Validation $G_{best}$', alpha=0.5)
            lines = lines + ln
            ln = axt.plot(vali_x_vals, raw_vali_bFits, '--', color=color, label='Validation $p_{best}$', alpha=0.6)
            lines = lines + ln

        # # add twin-axis combined legend
        labs = [l.get_label() for l in lines]
        plt.legend(lines, labs, loc=0)

        # Save figure for current loops best fitness variation
        fig_path = "%s/%d_Rep%d_FIG_Fit_V_%s.png" % (prm['SaveDir'], syst, rep, x)
        fig_evole.savefig(fig_path, dpi=150)
        plt.close(fig_evole)

    return


#

#

def AccuracyEvo(x, prm, syst, rep, training_err, TestFit, Global_bErrs, BatchPerEpoch,
                raw_vali_bErrs, batch_n_comp_list, epoch_n_comp_list, num_batches):
    """
    Plots A Evolution of the accuracy (i.e. data error)
    Can select the x axis to be Ncomps or Epochs.
    """

    if BatchPerEpoch == 0:
        batch_weight = 1
    else:
        batch_weight = 1/BatchPerEpoch

    if x == 'Epoch':
        new_stop = batch_weight*(len(training_err)-1)
        nspaces = len(training_err)
        train_x_vals = np.linspace(0, new_stop, nspaces)

        vali_x_vals = np.arange(1, len(raw_vali_bErrs)+1, 1)

    elif x == 'Ncomps':
        train_x_vals = batch_n_comp_list
        vali_x_vals = epoch_n_comp_list

    # # Don't Plot if no epochs where carried out
    if prm['DE']['epochs'] == 0:
        print("Won't plot FitEvo, as no epochs where carried out")
        return

    if prm['DE']['save_fitvit'] == 1:
        lines = []

        fig_evole = plt.figure(tight_layout=False)
        axes_evole = fig_evole.add_axes([0.125, 0.125, 0.75, 0.75])

        # Plot train pop best fit
        color = 'tab:blue'
        training_acc = 1 - np.array(training_err)
        ln1 = axes_evole.plot(train_x_vals, training_acc, '--', color=color, label='Train $p_{best}$', alpha=0.6)  # NOTE: This includes the best fitness of original population
        lines = ln1

        # Add trend line to train pop best fit
        if prm['DE']['batch_size'] != 0 and prm['DE']['batch_size'] != 1:
            av = moving_convolution(training_acc, N=num_batches)
            ln2 = axes_evole.plot(train_x_vals, av, '-', color=color, label='Training trend (convolution)', alpha=0.85)
            lines = lines + ln2

        if x == 'Epoch':
            if prm['DE']['batch_size'] == 0:
                axes_evole.set_xlabel('Generation')
            elif prm['DE']['batch_size'] == 1:
                axes_evole.set_xlabel('Epoch (Batch Size = 1 * TrainingData))')
            elif prm['DE']['batch_size'] < 1:
                axes_evole.set_xlabel('Epoch (Batch size = %.3f * TrainingData)' % (prm['DE']['batch_size']))
            else:
                axes_evole.set_xlabel('Epoch (Batch size = %d )' % (prm['DE']['batch_size']))
        elif x == 'Ncomps':
            if prm['DE']['batch_size'] == 0:
                axes_evole.set_xlabel('Ncomps')
            elif prm['DE']['batch_size'] == 1:
                axes_evole.set_xlabel('Ncomps (Batch Size = 1 * TrainingData))')
            elif prm['DE']['batch_size'] < 1:
                axes_evole.set_xlabel('Ncomps (Batch size = %.3f * TrainingData)' % (prm['DE']['batch_size']))
            else:
                axes_evole.set_xlabel('Ncomps (Batch size = %d )' % (prm['DE']['batch_size']))

        axes_evole.set_ylabel('Accuracy')
        #axes_evole.set_xlim([0, train_x_vals[-1]+1])

        # set tital
        new_title = 'Accuracy Evolution\n Test Error = %.4f' % (TestFit)
        axes_evole.set_title(new_title)

        # # Plot Validation accuracy
        if prm['DE']['batch_size'] != 0:

            color = 'tab:red'
            vali_acc = 1 - np.array(Global_bErrs)
            raw_vali_acc = 1 - np.array(raw_vali_bErrs)

            ln = axes_evole.plot(vali_x_vals, vali_acc, '-*', color=color, label='Validation $G_{best}$', alpha=0.5)
            lines = lines + ln
            ln = axes_evole.plot(vali_x_vals, raw_vali_acc, '--', color=color, label='Validation $p_{best}$', alpha=0.6)
            lines = lines + ln

        # # add twin-axis combined legend
        labs = [l.get_label() for l in lines]
        plt.legend(lines, labs, loc=0)

        # Save figure for current loops best fitness variation
        fig_path = "%s/%d_Rep%d_FIG_Accr_V_%s.png" % (prm['SaveDir'], syst, rep, x)
        fig_evole.savefig(fig_path, dpi=150)
        plt.close(fig_evole)

    return
#

#

#

#

#

#

def PCA_entropy(Vout):
    """
    Uses PCA to determin the entropy.

    Aim is that it is an indicator of how well "spread out" the Vout's are.
    (Unfortunatly no link between entopy & performance has been found)
    """

    # calc PCA entopy
    scaled_data = preprocessing.scale(Vout)
    pca = PCA() # create a PCA object
    pca.fit(scaled_data) # do the math
    eigenvalues_raw = pca.explained_variance_
    eigenvalues = eigenvalues_raw/np.max(eigenvalues_raw)
    entropy = 0
    for landa in eigenvalues:
        entropy = entropy - (landa*np.log(landa))

    # calc max entopy
    max_entropy = np.log(len(eigenvalues_raw))

    return entropy, max_entropy

#

#

#

def cluster2d(prm, syst, rep, lobj, the_data, best_responceY_list):
    """

    """

    data_X, data_y = lobj.get_data(the_data, iterate=0)  # load all data

    if len(data_X[0, :] == 2):

        model = cluster.KMeans(prm['DE']['num_classes'])
        model.fit(data_X)
        yhat_og = model.predict(data_X) # assign a cluster to each example
        og_com = met.completeness_score(data_y, yhat_og)



        fig, ax = plt.subplots(ncols=3, nrows=1, sharey=True, sharex=True)

        ax[0].scatter(data_X[:,0], data_X[:,1], c=data_y)
        ax[0].set_title('OG Data')
        ax[0].set_xlabel('a1')
        ax[0].set_ylabel('a2')
        ax[0].set_aspect('equal', adjustable='box')

        ax[1].scatter(data_X[:,0], data_X[:,1], c=yhat_og)
        ax[1].set_title('Clusters On OG')
        ax[1].set_xlabel('a1')
        ax[1].set_aspect('equal', adjustable='box')

        ax[2].scatter(data_X[:,0], data_X[:,1], c=best_responceY_list[-1])
        ax[2].set_title('Predicted Clusters')
        ax[2].set_xlabel('a1')
        ax[2].set_aspect('equal', adjustable='box')

        fig.suptitle('IntpScheme: %s, FitScheme: %s.' % (prm['DE']['IntpScheme'], prm['DE']['FitScheme']))


        fig0_path = "%s/%d_Rep%d_FIG_Cluster.png" % (prm['SaveDir'], syst, rep)
        fig.savefig(fig0_path, dpi=300)
        plt.close(fig)

    return

#

#

#

def plot_TransformedOutput(prm, syst, rep, lobj, the_data, best_responceY_list):
    """

    """

    data_X, data_y = lobj.get_data(the_data, iterate=0)  # load all data

    rY = best_responceY_list[-1]

    if len(np.shape(rY)) == 1:
        rY = np.reshape(rY, (-1, 1))


    if len(rY[0, :]) == 2:

        fig, ax = plt.subplots()
        ax.scatter(rY[:,0], rY[:,1], c=data_y)
        ax.set_title('Output')
        ax.set_xlabel('$op_1$')
        ax.set_ylabel('$op_2$')
        fig.suptitle('IntpScheme: %s, FitScheme: %s.' % (prm['DE']['IntpScheme'], prm['DE']['FitScheme']))
        fig0_path = "%s/%d_Rep%d_FIG_TransformedOutput.png" % (prm['SaveDir'], syst, rep)
        fig.savefig(fig0_path, dpi=300)
        plt.close(fig)

    elif len(rY[0, :]) == 1:

        fig, ax = plt.subplots()
        ax.scatter(np.arange(len(rY)), rY, c=data_y)
        ax.set_title('Output')
        ax.set_xlabel('instance')
        ax.set_ylabel('$op_1$')
        fig.suptitle('IntpScheme: %s, FitScheme: %s.' % (prm['DE']['IntpScheme'], prm['DE']['FitScheme']))
        fig0_path = "%s/%d_Rep%d_FIG_TransformedOutput.png" % (prm['SaveDir'], syst, rep)
        fig.savefig(fig0_path, dpi=300)
        plt.close(fig)

    return

#

#

#

def perfomance_comparison(prm, syst, rep, lobj, b_feature, the_data):
    """
    Performs Kmeans Clustering on the original data, on the provided features,
    and on the PCA data (using the same number of components as features).

    If 2d then a plot is produced, else text results are outputted.
    """

    # get training data
    data_X, data_Y = lobj.fetch_data(the_data)
    nfeatues = len(b_feature[0, :])

    # # apply clustering to original training data
    model_og = cluster.KMeans(prm['DE']['num_classes']) # , n_jobs=1
    model_og.fit(data_X)
    yhat_og = model_og.predict(data_X) # assign a cluster to each example
    #print("yhat:\n", yhat)
    raw_com, raw_homo, raw_V = met.homogeneity_completeness_v_measure(data_Y, yhat_og)
    report = ">> KMeans Cluster Performance <<\n"
    report += "Raw Training Data Vscore: %f\n" % (raw_V)


    # # Cluster on Features
    # apply clustering to original training data
    model_f = cluster.KMeans(prm['DE']['num_classes']) # , n_jobs=1
    model_f.fit(b_feature)
    yhat_f = model_f.predict(b_feature) # assign a cluster to each example
    #print("yhat_og:\n", yhat_og)
    f_com, f_homo, f_V = met.homogeneity_completeness_v_measure(data_Y, yhat_f)
    report += "Output Data Vscore: %f\n" % (f_V)


    # # Apply PCA to training data
    scaled_data = preprocessing.scale(data_X)
    pca = PCA() # create a PCA object
    pca.fit(scaled_data) # do the math
    pca_data = pca.transform(scaled_data) # PCA data (array)

    # # Cluster the PCA Data
    model_pca = cluster.KMeans(prm['DE']['num_classes']) # , n_jobs=1
    model_pca.fit(pca_data)
    yhat_pca = model_pca.predict(pca_data) # assign a cluster to each example
    pca_com, pca_homo, pca_V = met.homogeneity_completeness_v_measure(data_Y, yhat_pca)
    report += "PCA Component Vscore: %f\n" % (pca_V)

    print(report)
    del model_og, model_f, model_pca

    # # Make Plot or Report and save
    if nfeatues == 2:

        fig, ax = plt.subplots(ncols=3, nrows=1, sharey=True, sharex=True, figsize=(10, 6))

        col = np.where(data_Y==1,'b', np.where(data_Y==2,'r',np.where(data_Y==3,'g','k')))
        ax[0].scatter(b_feature[:,0], b_feature[:,1], c=col, s=4, alpha=0.6)
        ax[0].set_title('BFeature (real Classes)\nRaw Data: KMean Comp=%f\nRaw Data: KMean Homo=%f\nRaw Data: KMean Vscore=%f' % (raw_com, raw_homo, raw_V))
        ax[0].set_xlabel('Op 1')
        ax[0].set_ylabel('Op 2')
        #ax[0].set_aspect('equal', adjustable='box')

        ax[1].scatter(b_feature[:,0], b_feature[:,1], c=yhat_og, s=4, alpha=0.6)
        ax[1].set_title('BFeature (KMeans Prediction)\nKMean Comp=%f\nKMean Homo=%f\nKMean Vscore=%f' % (f_com, f_homo, f_V))
        ax[1].set_xlabel('Op 1')
        ax[1].set_ylabel('Op 2')
        #ax[1].set_aspect('equal', adjustable='box')

        ax[2].scatter(pca_data[:,0], pca_data[:,1], c=yhat_pca, s=4, alpha=0.6)
        ax[2].set_title('PCA (2 components)\nKMean Comp=%f\nKMean Homo=%f\nKMean Vscore=%f' % (pca_com, pca_homo, pca_V))
        ax[2].set_xlabel('component 1')
        ax[2].set_ylabel('component 2')
        #ax[2].set_aspect('equal', adjustable='box')

        fig.suptitle('OG Data  vs  EiM Outputs  vs  PCA')

        plt.subplots_adjust(top=0.75)

        fig0_path = "%s/%d_Rep%d_FIG_ClusterPerformance.png" % (prm['SaveDir'], syst, rep)
        fig.savefig(fig0_path, dpi=300)
        plt.close(fig)

    else:
        path_deets = "%s/%d_Rep%d_Report_ClusterPerformance.txt" % (prm['SaveDir'], syst, rep)
        file1 = open(path_deets, "w")
        file1.write(report)
        file1.close()

    return

#

#

#

def plot_InToOps(prm, syst, rep, other_InNodes='float'):
    """

    """

    num_in = prm['network']['num_input'] + prm['network']['num_config']
    axs_list = []
    fig_list = []

    if other_InNodes == 'float':
        other = '_float'
        other_text = 'floating'
    else:
        other = "%.2fV" % (other_InNodes)
        other_text = "%.2fV" % (other_InNodes)

    for o in range(prm['network']['num_output']):
        fig, ax = plt.subplots()
        ax.set_xlabel('Input Sweep [V]')
        ax.set_ylabel('$op_%d$ [V]' % (o+1))
        fig.suptitle('Output %d node (Other nodes @%s)' % (o+1, other_text))

        axs_list.append(ax)
        fig_list.append(fig)


    for i in range(num_in):
        inp = "in%d_conn" % (i+1)
        in_sweep, op_list = LoadRun_DC_InNodeToOps(prm, syst, n1=inp, other_InNodes=other_InNodes)

        for o in range(len(op_list)):
            axs_list[o].plot(in_sweep, op_list[o], label="in%d" % (i+1))

    for o in range(prm['network']['num_output']):
        plt.legend()
        fig_path = "%s/%d_Rep%d_FIG_RawInputSweep_Op%d_OtherNodes%s.png" % (prm['SaveDir'], syst, rep, o+1, other)
        fig_list[o].savefig(fig_path, dpi=300)
    plt.close('all')

    return

#

#

#

def plot_TransConductance(prm, syst, rep, StaticVin2=1, ShuntR=0):
    """

    """

    num_in = prm['network']['num_input'] + prm['network']['num_config']
    axs_list = []
    fig_list = []

    fig, ax = plt.subplots()
    ax.set_xlabel('Input Sweep [V]')
    ax.set_ylabel('$op_1$ [A]')
    fig.suptitle('TransConductance (in2@%.2f, ShuntR@%.1fKohm) as In1 Varies' % (StaticVin2, ShuntR))

    ll = ['float', -1, 0, 1]
    ll = [-1, 0, 1]
    for oin in ll:

        in_sweep, op_i = LoadRun_DC_TransConductance(prm, syst, other_InNodes=oin, StaticVin2=StaticVin2, ShuntR=ShuntR)

        ax.plot(in_sweep, op_i, label="Other Nodes %s" % str(oin))


    plt.legend()
    fig_path = "%s/%d_Rep%d_FIG_TransConductance_Shunt%d.png" % (prm['SaveDir'], syst, rep, ShuntR)
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    return


























#

# fin
