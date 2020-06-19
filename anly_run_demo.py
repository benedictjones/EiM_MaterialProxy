from Module_Analysis.Analysis import analysis

"""
To examine any particular single run or experiment and produce plots from the saved data we can use anly_run.py.
The target folder can be set to either:
- A results folder (e.g *'Results/2020_06_17/__13_02_22__con2DDS__EXP_Demo_Exp_D_RN'*)
- An experiment folder (e.g *'Experiment_List/con2DDS/2020_06_17__13_02_22___EXP_Demo_Exp_D_RN'*)
"""

# select file to lead data from

demo_exp = 'Experiment_List/con2DDS/2020_06_17__13_59_18___EXP_Demo_Exp_D_RN'

demo_exp_Shuffle_Off = 'Results/2020_06_17/__13_59_18__con2DDS__EXP_Demo_Exp_D_RN'


obj_a = analysis(demo_exp, save_loc='Demo/anly_run_demo', format='png')
save = 1  # don't save to file, produce python figure



# Plot graphs to do with fitness convergence etc.
obj_a.Plt_basic(sel_dict='na', Save_NotShow=save, show=1, fill=1, ExpErrors=1,
                 StandardError=1, logplot=0,
                 HistMin='auto', HistMax='auto',
                 PlotOnly='all')


# Plotting surface plots
obj_a.Plt_mg(Save_NotShow=save, Bgeno=1, Dgeno=1, VC=0, VP=0, VoW=0, ViW=0,
             #
             titles='off'#,
             #
             #Specific_Cir_Rep=[1,1]  # select a specific [circuit, repetition] to plot
             )



#

# fin
