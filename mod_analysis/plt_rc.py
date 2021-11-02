import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import h5py
from mod_analysis.Set_Load_meta import LoadMetaData


class rc_plotter(object):

    def __init__(self, dir, the_data='test'):
        """
        Initialise object
        """
        self.dir = dir
        self.the_data = the_data
        self.CompiledDict = LoadMetaData(self.dir)

        # # load data
        if self.the_data == 'test':
            fit = 'fit'
            vop = 'Vout'
            H = 'entropy'
        else:
            fit = 'best_fitness'
            vop = 'best_Vout'
            H = 'entropy'

        location = "%s/data.hdf5" % (self.dir)
        hdf = h5py.File(location, 'r')
        self.Vfactor_matrix = np.zeros((self.CompiledDict['num_repetitions'], self.CompiledDict['num_systems']))
        self.fitness_matrix = np.zeros((self.CompiledDict['num_repetitions'], self.CompiledDict['num_systems']))
        self.entropy_matrix = np.zeros((self.CompiledDict['num_repetitions'], self.CompiledDict['num_systems']))
        self.Vo_list = []
        for sys in range(self.CompiledDict['num_systems']):
            for rep in range(self.CompiledDict['num_repetitions']):
                de_saved = hdf.get('/%d_rep%d/DE_data/%s' % (sys, rep, self.the_data))
                self.fitness_matrix[rep, sys] = np.array(de_saved.get(fit))
                self.entropy_matrix[rep, sys] = np.array(de_saved.get(H))
                Vo = np.array(de_saved.get(vop))
                self.Vo_list.append(Vo)
                self.Vfactor_matrix[rep, sys] = np.mean(np.std(Vo, axis=1))

        return


    def FitvH(self, save=0):
        print("Producing Fitness vs Entropy plots")

        # # scatter Plot
        figH = plt.figure()
        ax = figH.add_axes([0.125, 0.125, 0.75, 0.75])
        """for rep in range(self.CompiledDict['num_repetitions']):
            ax.scatter(self.fitness_matrix[rep,:], self.entropy_matrix[rep,:], marker='x')"""
        for sys in range(self.CompiledDict['num_systems']):
            ax.scatter(self.fitness_matrix[:,sys], self.entropy_matrix[:,sys], marker='x')


        fit_flat = self.fitness_matrix.flatten()
        H_flat = self.entropy_matrix.flatten()
        #grad, bias = np.polyfit(fit_flat, H_flat, 1)
        #ax.plot(fit_flat, fit_flat*grad + bias, '--k')
        ax.plot(np.unique(fit_flat), np.poly1d(np.polyfit(fit_flat, H_flat, 1))(np.unique(fit_flat)), '--k')

        ax.set_title('Scatter Plot of %s Fitness Vs Entropy (for all Sys & Rep)' % (self.the_data))
        ax.set_xlabel('Fitness')
        ax.set_ylabel('Entropy')
        fig_path = "%s/FIG_FitvH_%s.png" % (self.dir, self.the_data)
        if save == 1:
            figH.savefig(fig_path, dpi=300)
            plt.close(figH)

        # # 2d histogram plot
        figH2 = plt.figure()
        ax2 = figH2.add_axes([0.125, 0.125, 0.75, 0.75])
        fit_flat = self.fitness_matrix.flatten()
        H_flat = self.entropy_matrix.flatten()
        ax2.hist2d(fit_flat, H_flat, bins=(50, 50), cmap=plt.cm.BuPu) # BuPu
        ax2.set_title('2D Histogram of %s Fitness Vs Entropy (for all Sys & Rep)' % (self.the_data))
        ax2.set_xlabel('Fitness')
        ax2.set_ylabel('Entropy')
        fig_path = "%s/FIG_hist_FitvH_%s.png" % (self.dir, self.the_data)
        if save == 1:
            figH2.savefig(fig_path, dpi=300)
            plt.close(figH2)

        # Show all
        if save != 1:
            plt.show()
            plt.close('all')

        return

    #

    #

    #

    def FitvStdVo(self, save=0):
        print("Producing Fitness vs Mean Std OutputVoltages plots")


        # # scatter Plot
        figH = plt.figure()
        ax = figH.add_axes([0.125, 0.125, 0.75, 0.75])
        """for rep in range(self.CompiledDict['num_repetitions']):
            ax.scatter(self.fitness_matrix[rep,:], self.Vfactor_matrix[rep,:], marker='x')"""
        for sys in range(self.CompiledDict['num_systems']):
            ax.scatter(self.fitness_matrix[:,sys], self.Vfactor_matrix[:,sys], marker='x')

        fit_flat = self.fitness_matrix.flatten()
        Vfactor_flat = self.Vfactor_matrix.flatten()
        ax.plot(np.unique(fit_flat), np.poly1d(np.polyfit(fit_flat, Vfactor_flat, 1))(np.unique(fit_flat)), '--k')

        ax.set_title('Scatter Plot of %s Fitness Vs Mean Std OutputVoltages (for all Sys & Rep)' % (self.the_data))
        ax.set_xlabel('Fitness')
        ax.set_ylabel('Mean Std OutputVoltages')
        fig_path = "%s/FIG_FitvStdVo_%s.png" % (self.dir, self.the_data)
        if save == 1:
            figH.savefig(fig_path, dpi=300)
            plt.close(figH)

        # # 2d histogram plot
        figH2 = plt.figure()
        ax2 = figH2.add_axes([0.125, 0.125, 0.75, 0.75])
        fit_flat = self.fitness_matrix.flatten()
        Vfactor_flat = self.Vfactor_matrix.flatten()
        ax2.hist2d(fit_flat, Vfactor_flat, bins=(50, 50), cmap=plt.cm.BuPu) # BuPu

        ax2.set_title('2D Histogram of %s Fitness Vs Mean Std OutputVoltages (for all Sys & Rep)' % (self.the_data))
        ax2.set_xlabel('Fitness')
        ax2.set_ylabel('Mean Std OutputVoltages')
        fig_path = "%s/FIG_hist_FitvStdVo_%s.png" % (self.dir, self.the_data)
        if save == 1:
            figH2.savefig(fig_path, dpi=300)
            plt.close(figH2)

        # Show all
        if save != 1:
            plt.show()
            plt.close('all')

        return

    #

    #

    #

    def HvStdVo(self, save=0):
        print("Producing Fitness vs Mean Std OutputVoltages plots")

        self.CompiledDict = LoadMetaData(self.dir)
        ParamDict = self.CompiledDict['DE']
        NetworkDict = self.CompiledDict['network']
        GenomeDict = self.CompiledDict['genome']


        # # scatter Plot
        figH = plt.figure()
        ax = figH.add_axes([0.125, 0.125, 0.75, 0.75])
        """for rep in range(self.CompiledDict['num_repetitions']):
            ax.scatter(self.Vfactor_matrix[rep,:], self.entropy_matrix[rep,:], marker='x')"""
        for sys in range(self.CompiledDict['num_systems']):
            ax.scatter(self.Vfactor_matrix[:,sys], self.entropy_matrix[:,sys], marker='x')

        H_flat = self.entropy_matrix.flatten()
        Vfactor_flat = self.Vfactor_matrix.flatten()
        ax.plot(np.unique(Vfactor_flat), np.poly1d(np.polyfit(Vfactor_flat, H_flat, 1))(np.unique(Vfactor_flat)), '--k')

        ax.set_title('Scatter Plot of %s Mean Std OutputVoltages Vs Entropy (for all Sys & Rep)' % (self.the_data))
        ax.set_xlabel('Mean Std OutputVoltages')
        ax.set_ylabel('Entropy')
        fig_path = "%s/FIG_HvStdVo_%s.png" % (self.dir, self.the_data)
        if save == 1:
            figH.savefig(fig_path, dpi=300)
            plt.close(figH)

        # # 2d histogram plot
        figH2 = plt.figure()
        ax2 = figH2.add_axes([0.125, 0.125, 0.75, 0.75])
        H_flat = self.entropy_matrix.flatten()
        Vfactor_flat = self.Vfactor_matrix.flatten()
        ax2.hist2d(Vfactor_flat, H_flat, bins=(50, 50), cmap=plt.cm.BuPu) # BuPu

        ax2.set_title('2D Histogram of %s Mean Std OutputVoltages Vs Entropy (for all Sys & Rep)' % (self.the_data))
        ax2.set_xlabel('Mean Std OutputVoltages')
        ax2.set_ylabel('Entropy')
        fig_path = "%s/FIG_hist_HvStdVo_%s.png" % (self.dir, self.the_data)
        if save == 1:
            figH2.savefig(fig_path, dpi=300)
            plt.close(figH2)

        # Show all
        if save != 1:
            plt.show()
            plt.close('all')

        return

    #

    #

    #

    def test(self, save=0):
        print("Producing Fitness vs Mean Std OutputVoltages plots")

        Vf_matrix = np.zeros((self.CompiledDict['num_repetitions'], self.CompiledDict['num_systems']))

        cont = 0
        for sys in range(self.CompiledDict['num_systems']):
            for rep in range(self.CompiledDict['num_repetitions']):
                Vo = self.Vo_list[cont]

                Vf_matrix[rep, sys] = np.mean(np.std(Vo, axis=1))
                Vf_matrix[rep, sys] = np.std(np.std(Vo, axis=1))
                Vf_matrix[rep, sys] = np.mean(np.mean(abs(Vo), axis=1))
                Vf_matrix[rep, sys] = np.std(np.max(Vo, axis=1)-np.min(Vo, axis=1))
                Vf_matrix[rep, sys] = np.mean(np.max(Vo, axis=1)-np.min(Vo, axis=1))
                cont = cont + 1

        # # scatter Plot
        figH = plt.figure()
        ax = figH.add_axes([0.125, 0.125, 0.75, 0.75])
        """for rep in range(self.CompiledDict['num_repetitions']):
            ax.scatter(self.fitness_matrix[rep,:], Vf_matrix[rep,:], marker='x')"""
        for sys in range(self.CompiledDict['num_systems']):
            ax.scatter(self.fitness_matrix[:,sys], Vf_matrix[:,sys], marker='x')

        fit_flat = self.fitness_matrix.flatten()
        Vfactor_flat = Vf_matrix.flatten()
        ax.plot(np.unique(fit_flat), np.poly1d(np.polyfit(fit_flat, Vfactor_flat, 1))(np.unique(fit_flat)), '--k')

        ax.set_title('Scatter Plot of %s Fitness Vs Mean Std OutputVoltages (for all Sys & Rep)' % (self.the_data))
        ax.set_xlabel('Fitness')
        ax.set_ylabel('Factor')
        fig_path = "%s/FIG_FitvStdVo_%s.png" % (self.dir, self.the_data)
        if save == 1:
            figH.savefig(fig_path, dpi=300)
            plt.close(figH)

        #"""
        # # 2d histogram plot
        figH2 = plt.figure()
        ax2 = figH2.add_axes([0.125, 0.125, 0.75, 0.75])
        fit_flat = self.fitness_matrix.flatten()
        Vfactor_flat = Vf_matrix.flatten()
        ax2.hist2d(fit_flat, Vfactor_flat, bins=(50, 50), cmap=plt.cm.BuPu) # BuPu

        ax2.set_title('2D Histogram of %s Fitness Vs Mean Std OutputVoltages (for all Sys & Rep)' % (self.the_data))
        ax2.set_xlabel('Fitness')
        ax2.set_ylabel('Mean Std OutputVoltages')
        fig_path = "%s/FIG_hist_FitvStdVo_%s.png" % (self.dir, self.the_data)
        if save == 1:
            figH2.savefig(fig_path, dpi=300)
            plt.close(figH2)
        #"""

        # Show all
        if save != 1:
            plt.show()
            plt.close('all')

        return

#

# fin
