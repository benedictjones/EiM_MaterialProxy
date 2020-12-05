from mod_analysis.Analysis import analysis




######################################
# # Paper data locations


# # 2DDS
a_RRN = 'Experiment_List/2DDS/2020_11_18__21_56_19___EXP_PaperReplication__R_RN'
a_DRN = 'Experiment_List/2DDS/2020_11_18__22_14_46___EXP_PaperReplication__D_RN'
a_NL = 'Experiment_List/2DDS/2020_11_18__22_53_47___EXP_PaperReplication__NL_RN'


# # con2DDS
b_RRN = 'Experiment_List/con2DDS/2020_11_19__09_45_05___EXP_PaperReplication__R_RN'
b_DRN = 'Experiment_List/con2DDS/2020_11_19__12_28_36___EXP_PaperReplication__D_RN'
b_NL = 'Experiment_List/con2DDS/2020_11_19__15_01_55___EXP_PaperReplication__NL_RN'


# # flipped2DDS
c_RRN = 'Experiment_List/flipped_2DDS/2020_11_20__09_10_58___EXP_PaperReplication__R_RN'
c_DRN = 'Experiment_List/flipped_2DDS/2020_11_20__09_59_16___EXP_PaperReplication__D_RN'
c_NL = 'Experiment_List/flipped_2DDS/2020_11_20__11_11_49___EXP_PaperReplication__NL_RN'


# ###########################################
# # Generate plots for paper
# ###########################################


# # Fitness Plots
save = 1
form = 'png'  # or 'pdf' , 'png', 'svg'



#

#
ax=[0,0.15]
obj_a = analysis(a_RRN, save_loc='Paper/graphs/_2dds_r_all', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_basic(sel_dict='na', Save_NotShow=save, show=1, fill=1, ExpErrors=1,
                 StandardError=1,
                 #logplot=1, legends='on', plt_mean_yaxis=[0.001,0.5],
                 logplot=0, legends='off', plt_mean_yaxis=ax,
                 HistMin='auto', HistMax='auto',
                 fig_letter='a',
                 PlotOnly='all', titles='off')  # HistBins=0.02, HistMin=0, HistMax='auto' PlotOnly=[3+5,4+5]

obj_a = analysis(a_RRN, save_loc='Paper/graphs/_2dds_r_main', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_basic(sel_dict='na', Save_NotShow=save, show=1, fill=1, ExpErrors=1,
                 StandardError=1, logplot=0,
                 HistMin='auto', HistMax='auto',
                 fig_letter='a',
                 PlotOnly=[0,1,2,4,6], titles='off', plt_mean_yaxis=ax)  #

obj_a = analysis(a_RRN, save_loc='Paper/graphs/_2dds_r_sup', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_basic(sel_dict='na', Save_NotShow=save, show=1, fill=1, ExpErrors=1,
                 StandardError=1, logplot=0,
                 fig_letter='a',
                 HistMin='auto', HistMax='auto',
                 PlotOnly=[0,3,5,7], titles='off', plt_mean_yaxis=ax)

#

#

obj_a = analysis(a_DRN, save_loc='Paper/graphs/_2dds_d_all', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_basic(sel_dict='na', Save_NotShow=save, show=1, fill=1, ExpErrors=1,
                 StandardError=1,
                 #logplot=1, legends='on', plt_mean_yaxis=[0.001,0.5],
                 logplot=0, legends='off', plt_mean_yaxis=ax,
                 HistMin='auto', HistMax='auto',
                 fig_letter='c',
                 PlotOnly='all', titles='off')  # HistBins=0.02, HistMin=0, HistMax='auto' PlotOnly=[3+5,4+5]

obj_a = analysis(a_DRN, save_loc='Paper/graphs/_2dds_d_main', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_basic(sel_dict='na', Save_NotShow=save, show=1, fill=1, ExpErrors=1,
                 StandardError=1, logplot=0,
                 HistMin='auto', HistMax='auto',
                 fig_letter='c',
                 PlotOnly=[0,1,2,4,6], titles='off', plt_mean_yaxis=ax)  #

obj_a = analysis(a_DRN, save_loc='Paper/graphs/_2dds_d_sup', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_basic(sel_dict='na', Save_NotShow=save, show=1, fill=1, ExpErrors=1,
                 StandardError=1, logplot=0,
                 HistMin='auto', HistMax='auto',
                 fig_letter='c',
                 PlotOnly=[0,3,5,7], titles='off', plt_mean_yaxis=ax)

#

#

obj_a = analysis(a_NL, save_loc='Paper/graphs/_2dds_nl_all', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_basic(sel_dict='na', Save_NotShow=save, show=1, fill=1, ExpErrors=1,
                 StandardError=1,
                 #logplot=1, legends='on', plt_mean_yaxis=[0.001,0.5],
                 logplot=0, legends='off', plt_mean_yaxis=ax,
                 HistMin='auto', HistMax='auto',
                 fig_letter='b',
                 PlotOnly='all', titles='off')  # HistBins=0.02, HistMin=0, HistMax='auto' PlotOnly=[3+5,4+5]

obj_a = analysis(a_NL, save_loc='Paper/graphs/_2dds_nl_main', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_basic(sel_dict='na', Save_NotShow=save, show=1, fill=1, ExpErrors=1,
                 StandardError=1, logplot=0,
                 HistMin='auto', HistMax='auto',
                 fig_letter='b',
                 PlotOnly=[0,1,2,4,6], titles='off', plt_mean_yaxis=ax)  #

obj_a = analysis(a_NL, save_loc='Paper/graphs/_2dds_nl_sup', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_basic(sel_dict='na', Save_NotShow=save, show=1, fill=1, ExpErrors=1,
                 StandardError=1, logplot=0,
                 HistMin='auto', HistMax='auto',
                 fig_letter='b',
                 PlotOnly=[0,3,5,7], titles='off', plt_mean_yaxis=ax)

#

#
exit()
#

#

#

#

obj_a = analysis(b_RRN, save_loc='Paper/graphs/_con2dds_r_all', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_basic(sel_dict='na', Save_NotShow=save, show=1, fill=1, ExpErrors=1,
                 StandardError=1,
                 #logplot=1, legends='off', plt_mean_yaxis=[0.1,0.5],
                 logplot=0, legends='off', plt_mean_yaxis=[0.1,0.41],
                 HistMin='auto', HistMax='auto',
                 fig_letter='a',
                 PlotOnly='all', titles='off')  # HistBins=0.02, HistMin=0, HistMax='auto' PlotOnly=[3+5,4+5]

obj_a = analysis(b_RRN, save_loc='Paper/graphs/_con2dds_r_main', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_basic(sel_dict='na', Save_NotShow=save, show=1, fill=1, ExpErrors=1,
                 StandardError=1, logplot=0,
                 HistMin='auto', HistMax='auto',
                 fig_letter='a',
                 PlotOnly=[0,1,2,4,6], titles='off', plt_mean_yaxis=[0.1,0.41])  #

obj_a = analysis(b_RRN, save_loc='Paper/graphs/_con2dds_r_sup', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_basic(sel_dict='na', Save_NotShow=save, show=1, fill=1, ExpErrors=1,
                 StandardError=1, logplot=0,
                 HistMin='auto', HistMax='auto',
                 fig_letter='a',
                 PlotOnly=[0,3,5,7], titles='off', plt_mean_yaxis=[0.1,0.41])

#

#

obj_a = analysis(b_DRN, save_loc='Paper/graphs/_con2dds_d_all', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_basic(sel_dict='na', Save_NotShow=save, show=1, fill=1, ExpErrors=1,
                 StandardError=1,
                 #logplot=1, legends='off', plt_mean_yaxis=[0.1,0.5],
                 logplot=0, legends='off', plt_mean_yaxis=[0.1,0.41],
                 HistMin='auto', HistMax='auto',
                 fig_letter='c',
                 PlotOnly='all', titles='off')  # HistBins=0.02, HistMin=0, HistMax='auto' PlotOnly=[3+5,4+5]

obj_a = analysis(b_DRN, save_loc='Paper/graphs/_con2dds_d_main', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_basic(sel_dict='na', Save_NotShow=save, show=1, fill=1, ExpErrors=1,
                 StandardError=1, logplot=0,
                 HistMin='auto', HistMax='auto',
                 fig_letter='c',
                 PlotOnly=[0,1,2,4,6], titles='off', plt_mean_yaxis=[0.1,0.41])  #

obj_a = analysis(b_DRN, save_loc='Paper/graphs/_con2dds_d_sup', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_basic(sel_dict='na', Save_NotShow=save, show=1, fill=1, ExpErrors=1,
                 StandardError=1, logplot=0,
                 HistMin='auto', HistMax='auto',
                 fig_letter='c',
                 PlotOnly=[0,3,5,7], titles='off', plt_mean_yaxis=[0.1,0.41])

#

#

obj_a = analysis(b_NL, save_loc='Paper/graphs/_con2dds_nl_all', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_basic(sel_dict='na', Save_NotShow=save, show=1, fill=1, ExpErrors=1,
                 StandardError=1,
                 #logplot=1, legends='off', plt_mean_yaxis=[0.1,0.5],
                 logplot=0, legends='off', plt_mean_yaxis=[0.1,0.41],
                 HistMin='auto', HistMax='auto',
                 fig_letter='b',
                 PlotOnly='all', titles='off')  # HistBins=0.02, HistMin=0, HistMax='auto' PlotOnly=[3+5,4+5]

obj_a = analysis(b_NL, save_loc='Paper/graphs/_con2dds_nl_main', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_basic(sel_dict='na', Save_NotShow=save, show=1, fill=1, ExpErrors=1,
                 StandardError=1, logplot=0,
                 HistMin='auto', HistMax='auto',
                 fig_letter='b',
                 PlotOnly=[0,1,2,4,6], titles='off', plt_mean_yaxis=[0.1,0.41])  #

obj_a = analysis(b_NL, save_loc='Paper/graphs/_con2dds_nl_sup', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_basic(sel_dict='na', Save_NotShow=save, show=1, fill=1, ExpErrors=1,
                 StandardError=1, logplot=0,
                 HistMin='auto', HistMax='auto',
                 fig_letter='b',
                 PlotOnly=[0,3,5,7], titles='off', plt_mean_yaxis=[0.1,0.41])

#

#

#

#

#

#

#

#

obj_a = analysis(c_RRN, save_loc='Paper/graphs/_f2dds_r_all', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_basic(sel_dict='na', Save_NotShow=save, show=1, fill=1, ExpErrors=1,
                 StandardError=1,
                 #logplot=1, legends='off', plt_mean_yaxis=[0.01,0.6],
                 logplot=0, legends='off', plt_mean_yaxis=[0,0.52],
                 HistMin='auto', HistMax='auto',
                 fig_letter='a',
                 PlotOnly='all', titles='off')  # HistBins=0.02, HistMin=0, HistMax='auto' PlotOnly=[3+5,4+5]

obj_a = analysis(c_RRN, save_loc='Paper/graphs/_f2dds_r_main', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_basic(sel_dict='na', Save_NotShow=save, show=1, fill=1, ExpErrors=1,
                 StandardError=1, logplot=0,
                 HistMin='auto', HistMax='auto',
                 fig_letter='a',
                 PlotOnly=[0,1,2,4,6], titles='off', plt_mean_yaxis=[0,0.52])  #

obj_a = analysis(c_RRN, save_loc='Paper/graphs/_f2dds_r_sup', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_basic(sel_dict='na', Save_NotShow=save, show=1, fill=1, ExpErrors=1,
                 StandardError=1, logplot=0,
                 HistMin='auto', HistMax='auto',
                 fig_letter='a',
                 PlotOnly=[0,3,5,7], titles='off', plt_mean_yaxis=[0,0.52])

#

#

obj_a = analysis(c_DRN, save_loc='Paper/graphs/_f2dds_d_all', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_basic(sel_dict='na', Save_NotShow=save, show=1, fill=1, ExpErrors=1,
                 StandardError=1,
                 #logplot=1, legends='off', plt_mean_yaxis=[0.01,0.6],
                 logplot=0, legends='off', plt_mean_yaxis=[0,0.52],
                 HistMin='auto', HistMax='auto',
                 fig_letter='c',
                 PlotOnly='all', titles='off')  # HistBins=0.02, HistMin=0, HistMax='auto' PlotOnly=[3+5,4+5]

obj_a = analysis(c_DRN, save_loc='Paper/graphs/_f2dds_d_main', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_basic(sel_dict='na', Save_NotShow=save, show=1, fill=1, ExpErrors=1,
                 StandardError=1, logplot=0,
                 HistMin='auto', HistMax='auto',
                 fig_letter='c',
                 PlotOnly=[0,1,2,4,6], titles='off', plt_mean_yaxis=[0,0.52])  #

obj_a = analysis(c_DRN, save_loc='Paper/graphs/_f2dds_d_sup', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_basic(sel_dict='na', Save_NotShow=save, show=1, fill=1, ExpErrors=1,
                 StandardError=1, logplot=0,
                 HistMin='auto', HistMax='auto',
                 fig_letter='c',
                 PlotOnly=[0,3,5,7], titles='off', plt_mean_yaxis=[0,0.52])

#

#

obj_a = analysis(c_NL, save_loc='Paper/graphs/_f2dds_nl_all', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_basic(sel_dict='na', Save_NotShow=save, show=1, fill=1, ExpErrors=1,
                 StandardError=1,
                 #logplot=1, legends='off', plt_mean_yaxis=[0.01,0.6],
                 logplot=0, legends='off', plt_mean_yaxis=[0,0.52],
                 HistMin='auto', HistMax='auto',
                 fig_letter='b',
                 PlotOnly='all', titles='off')  # HistBins=0.02, HistMin=0, HistMax='auto' PlotOnly=[3+5,4+5]

obj_a = analysis(c_NL, save_loc='Paper/graphs/_f2dds_nl_main', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_basic(sel_dict='na', Save_NotShow=save, show=1, fill=1, ExpErrors=1,
                 StandardError=1, logplot=0,
                 HistMin='auto', HistMax='auto',
                 fig_letter='b',
                 PlotOnly=[0,1,2,4,6], titles='off', plt_mean_yaxis=[0,0.52])  #

obj_a = analysis(c_NL, save_loc='Paper/graphs/_f2dds_nl_sup', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_basic(sel_dict='na', Save_NotShow=save, show=1, fill=1, ExpErrors=1,
                 StandardError=1, logplot=0,
                 HistMin='auto', HistMax='auto',
                 fig_letter='b',
                 PlotOnly=[0,3,5,7], titles='off', plt_mean_yaxis=[0,0.52])

#

# fin
