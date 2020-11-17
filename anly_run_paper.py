from mod_analysis.Analysis import analysis




######################################
# # Paper data locations



VaryNL_expandn09 = 'Experiment_List/con2DDS/2020_11_04__11_57_59___EXP_VaryNL_Cov-0.9__NL_RN'
VaryNL_expand0 = 'Experiment_List/con2DDS/2020_11_04__12_35_00___EXP_VaryNL_Cov0__NL_RN'
VaryNL_expand09 = 'Experiment_List/con2DDS/2020_11_04__13_16_40___EXP_VaryNL_Cov0.9__NL_RN'

VaryNL_bandn09 = 'Experiment_List/con2DDS/2020_11_05__18_04_52___EXP_VaryNL_shiftband_Cov-0.9__NL_RN'
VaryNL_band0 = 'Experiment_List/con2DDS/2020_11_05__18_41_37___EXP_VaryNL_shiftband_Cov0__NL_RN'
VaryNL_band09 = 'Experiment_List/con2DDS/2020_11_05__19_22_02___EXP_VaryNL_shiftband_Cov0.9__NL_RN'

# ###########################################
# # Generate plots for paper
# ###########################################


# # Fitness Plots
save = 1
form = 'png'  # or 'pdf'

obj_a = analysis(VaryNL_expandn09, save_loc='VaryNL/expand/-09', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_basic(sel_dict='na', Save_NotShow=save, show=1, fill=1, ExpErrors=1,
                 StandardError=1, logplot=0,
                 HistMin='auto', HistMax='auto',
                 PlotOnly='all', titles='off', plt_mean_yaxis=[0.25,0.35])  # HistBins=0.02, HistMin=0, HistMax='auto' PlotOnly=[3+5,4+5]

obj_a = analysis(VaryNL_expand0, save_loc='VaryNL/expand/0', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_basic(sel_dict='na', Save_NotShow=save, show=1, fill=1, ExpErrors=1,
                 StandardError=1, logplot=0,
                 HistMin='auto', HistMax='auto',
                 titles='off', plt_mean_yaxis=[0.25,0.35])  #

obj_a = analysis(VaryNL_expand09, save_loc='VaryNL/expand/09', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_basic(sel_dict='na', Save_NotShow=save, show=1, fill=1, ExpErrors=1,
                 StandardError=1, logplot=0,
                 HistMin='auto', HistMax='auto',
                 titles='off', plt_mean_yaxis=[0.25,0.35])

#

#


#

obj_a = analysis(VaryNL_bandn09, save_loc='VaryNL/band/-09', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_basic(sel_dict='na', Save_NotShow=save, show=1, fill=1, ExpErrors=1,
                 StandardError=1, logplot=0,
                 HistMin='auto', HistMax='auto',
                 PlotOnly='all', titles='off', plt_mean_yaxis=[0.25,0.35])  # HistBins=0.02, HistMin=0, HistMax='auto' PlotOnly=[3+5,4+5]

obj_a = analysis(VaryNL_band0, save_loc='VaryNL/band/0', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_basic(sel_dict='na', Save_NotShow=save, show=1, fill=1, ExpErrors=1,
                 StandardError=1, logplot=0,
                 HistMin='auto', HistMax='auto',
                 titles='off', plt_mean_yaxis=[0.25,0.35])  #

obj_a = analysis(VaryNL_band09, save_loc='VaryNL/band/09', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_basic(sel_dict='na', Save_NotShow=save, show=1, fill=1, ExpErrors=1,
                 StandardError=1, logplot=0,
                 HistMin='auto', HistMax='auto',
                 titles='off', plt_mean_yaxis=[0.25,0.35])






#

# fin
