from mod_analysis.Analysis import analysis




######################################
# # Paper data locations




target = 'Experiment_List/con2DDS/2021_01_26__12_00_47___EXP_Batching__D_RN'

target = 'Results/2021_03_09/__11_46_50__con2DDS__EXP_BatchingFinal__D_RN__EiM'

target = 'Results_Experiments/con2DDS/2021_04_14__13_00_21___EXP_IntrpScheme__Fit_BinCrossEntropy__D_RN'


target = 'Results_Experiments/c2DDS/2021_05_11__09_55_57___EXP_FitScheme_OG_pn_binary__D_RN_withExtra'

#target = 'Results/2021_05_12/__10_27_32__BankNote__EXP_FitScheme_extra_pn_binary__D_RN__EiM'
target = 'Results_Experiments/BankNote/2021_05_12__12_00_00___EXP_FitScheme_CUSTOM_pn_binary__D_RN'

target = 'Results_Experiments/c2DDS/2021_05_13__13_07_43___EXP_VaryNodes__D_RN'

# ###########################################
# # Generate plots for paper
# ###########################################


target = 'Results_Experiments/c2DDS/2021_06_11__23_01_10___EXP_IntrpScheme__error__D_RN'

# # Fitness Plots
save = 1
form = 'png'  # or 'pdf'

sel_dict = {'plt_mean':1,'plt_std':0,'plt_finalveri':0,'plt_popmean':0,'plt_box':1,'plt_hist':0}
#sel_dict = 'na'

obj_a = analysis(target, save_loc=0, format=form)  # dir_NameList=FilePAram,
obj_a.Plt_basic(Save_NotShow=save, fill=1, ExpErrors=1, StandardError=1)

"""obj_a.Plt_basic(sel_dict=sel_dict, Save_NotShow=save, show=1, fill=1, ExpErrors=1,
                 StandardError=1, logplot=0,
                 HistMin='auto', HistMax='auto',
                 PlotOnly='all', titles='off', plt_mean_yaxis=[0.25,0.35])  # HistBins=0.02, HistMin=0, HistMax='auto' PlotOnly=[3+5,4+5]
"""
# """

#

#


#







#

# fin
