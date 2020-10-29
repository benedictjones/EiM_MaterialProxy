from mod_analysis.Analysis import analysis

"""
Load data saved from previous code executions.
"""


# Select a target folder, e.g.
target = 'Experiment_List/con2DDS/2020_06_17__13_59_18___EXP_Demo_Exp_D_RN'
target = 'Results/2020_10_29/__13_38_58__con2DDS__D_RN__EiM'

# create object
obj_a = analysis(target, save_loc='Test', format='png')  # dir_NameList=FilePAram,
save = 1

# Plot graphs to do with fitness convergence etc.
obj_a.Plt_basic(sel_dict='na', Save_NotShow=save, show=1, fill=1, ExpErrors=1,
                 StandardError=1, logplot=0,
                 HistMin='auto', HistMax='auto',
                 PlotOnly='all')  # HistBins=0.02, HistMin=0, HistMax='auto' PlotOnly=[3+5,4+5]


# Plotting surface plots
obj_a.Plt_mg(Save_NotShow=save,
             #
             Bgeno=1, Dgeno=0, VC=0, VP=0, VoW=0, ViW=0,
             VoW_ani=0, ViW_ani=0, VoiW_ani=0, VloW_ani=0, VliW_ani=0,
             #
             Specific_Cir_Rep=[0,0],
             #
             titles='on')

#

# fin
