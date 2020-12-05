from mod_analysis.Analysis import analysis




######################################
# # Paper data locations



VaryNL_1 = 'Experiment_List/con2DDS/2020_11_20__22_15_38___EXP_NLcorr__NL_RN__alim_10_100_'
VaryNL_2 = 'Experiment_List/con2DDS/2020_11_20__23_43_41___EXP_NLcorr__NL_RN__alim_10_1000_'
VaryNL_3 = 'Experiment_List/con2DDS/2020_11_21__01_24_31___EXP_NLcorr__NL_RN__alim_10_10000_'


# ###########################################
# # Generate plots for paper
# ###########################################


# # Fitness Plots
save = 1
form = 'png'  # or 'pdf'

sel_dict = {'plt_mean':0,'plt_std':0,'plt_finalveri':0,'plt_popmean':0,'plt_box':1,'plt_hist':0}
sel_dict = 'na'

obj_a = analysis(VaryNL_1, save_loc='VaryNL/alim_10_100', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_basic(sel_dict=sel_dict, Save_NotShow=save, show=1, fill=1, ExpErrors=1,
                 StandardError=1, logplot=0,
                 HistMin='auto', HistMax='auto',
                 PlotOnly='all', titles='off', plt_mean_yaxis=[0.25,0.35])  # HistBins=0.02, HistMin=0, HistMax='auto' PlotOnly=[3+5,4+5]

"""
obj_a = analysis(VaryNL_2, save_loc='VaryNL/alim_10_1000', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_basic(sel_dict='na', Save_NotShow=save, show=1, fill=1, ExpErrors=1,
                 StandardError=1, logplot=0,
                 HistMin='auto', HistMax='auto',
                 fig_letter='a',
                 titles='off', plt_mean_yaxis=[0.25,0.35])
#"""


obj_a = analysis(VaryNL_3, save_loc='VaryNL/alim_10_10000', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_basic(sel_dict=sel_dict, Save_NotShow=save, show=1, fill=1, ExpErrors=1,
                 StandardError=1, logplot=0,
                 HistMin='auto', HistMax='auto',
                 titles='off', plt_mean_yaxis=[0.25,0.35])
#"""
#

#


#







#

# fin
