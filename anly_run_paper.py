from Module_Analysis.Analysis import analysis




######################################
# # Paper data locations


# # The Experiment runs containing the training and verification fitness data
_2dds_r = 'Experiment_List/2DDS/2020_04_22__17_29_36___EXP_ForPaper_VaryScheme_R_RN'
_2dds_d = 'Experiment_List/2DDS/2020_04_25__21_29_21___EXP_ForPaper_VaryScheme_Attempt2_D_RN'

_con2dds_r = 'Experiment_List/con2DDS/2020_04_28__02_18_04___EXP_ForPaper_VaryScheme_R_RN'
_con2dds_d = 'Experiment_List/con2DDS/2020_05_01__08_57_01___EXP_ForPaper_VaryScheme_D_RN'


# # MG (network responce surface plot) data used in paper
DRN_shuffle = 'Results/2020_03_30/__17_03_46__2DDS__EXP_Generating_MG_2DDS_withIW'
RRN_shuffle = 'Results/2020_03_30/__16_49_09__2DDS__EXP_Generating_MG_2DDS_withIW'

RRN_config = 'Results/2020_03_30/__16_49_09__2DDS__EXP_Generating_MG_2DDS_withIW'
DRN_config = 'Results/2020_03_31/__18_44_54__con2DDS__EXP_Generating_MG_con2DDS_withIOW'

RRN_bestG = 'Results/2020_03_30/__11_09_28__2DDS__EXP_Generating_MG_2DDS'
DRN_defalt = 'Results/2020_03_30/__11_20_16__2DDS__EXP_Generating_MG_2DDS'
DRN_bestG = 'Results/2020_03_23/__17_09_44__2DDS__D_RN'

RRN_weights = 'Results/2020_03_31/__17_05_03__O2DDS__EXP_Generating_MG_O2DDS_withIOW'
DRN_weights = 'Results/2020_03_31/__18_44_54__con2DDS__EXP_Generating_MG_con2DDS_withIOW'



# ###########################################
# # Generate plots for paper
# ###########################################


# # Fitness Plots
save = 1
form = 'png'  # or 'pdf'

obj_a = analysis(_2dds_r, save_loc='Paper/graphs/_2dds_r_all', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_basic(sel_dict='na', Save_NotShow=save, show=1, fill=1, ExpErrors=1,
                 StandardError=1, logplot=0,
                 HistMin='auto', HistMax='auto',
                 PlotOnly='all', titles='off')  # HistBins=0.02, HistMin=0, HistMax='auto' PlotOnly=[3+5,4+5]

obj_a = analysis(_2dds_r, save_loc='Paper/graphs/_2dds_r_main', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_basic(sel_dict='na', Save_NotShow=save, show=1, fill=1, ExpErrors=1,
                 StandardError=1, logplot=0,
                 HistMin='auto', HistMax='auto',
                 PlotOnly=[0,1,2,4,6], titles='off', plt_mean_yaxis=[0,0.35])  #

obj_a = analysis(_2dds_r, save_loc='Paper/graphs/_2dds_r_sup', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_basic(sel_dict='na', Save_NotShow=save, show=1, fill=1, ExpErrors=1,
                 StandardError=1, logplot=0,
                 HistMin='auto', HistMax='auto',
                 PlotOnly=[0,3,5,7], titles='off', plt_mean_yaxis=[0,0.38])

#

#

obj_a = analysis(_2dds_d, save_loc='Paper/graphs/_2dds_d_all', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_basic(sel_dict='na', Save_NotShow=save, show=1, fill=1, ExpErrors=1,
                 StandardError=1, logplot=0,
                 HistMin='auto', HistMax='auto',
                 PlotOnly='all', titles='off')  # HistBins=0.02, HistMin=0, HistMax='auto' PlotOnly=[3+5,4+5]


obj_a = analysis(_2dds_d, save_loc='Paper/graphs/_2dds_d_main', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_basic(sel_dict='na', Save_NotShow=save, show=1, fill=1, ExpErrors=1,
                 StandardError=1, logplot=0,
                 HistMin='auto', HistMax='auto',
                 PlotOnly=[0,1,2,4,6], titles='off', plt_mean_yaxis=[0,0.35])  #

obj_a = analysis(_2dds_d, save_loc='Paper/graphs/_2dds_d_sup', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_basic(sel_dict='na', Save_NotShow=save, show=1, fill=1, ExpErrors=1,
                 StandardError=1, logplot=0,
                 HistMin='auto', HistMax='auto',
                 PlotOnly=[0,3,5,7], titles='off', plt_mean_yaxis=[0,0.38])

#

#

obj_a = analysis(_con2dds_r, save_loc='Paper/graphs/_con2dds_r_all', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_basic(sel_dict='na', Save_NotShow=save, show=1, fill=1, ExpErrors=1,
                 StandardError=1, logplot=0,
                 HistMin=0, HistMax='auto',
                 PlotOnly='all', titles='off')  # HistBins=0.02, HistMin=0, HistMax='auto' PlotOnly=[3+5,4+5]

obj_a = analysis(_con2dds_r, save_loc='Paper/graphs/_con2dds_r_main', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_basic(sel_dict='na', Save_NotShow=save, show=1, fill=1, ExpErrors=1,
                 StandardError=1, logplot=0,
                 HistMin='auto', HistMax='auto',
                 PlotOnly=[0,1,2,4,6], titles='off', plt_mean_yaxis=[0.2,0.42])  #

obj_a = analysis(_con2dds_r, save_loc='Paper/graphs/_con2dds_r_sup', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_basic(sel_dict='na', Save_NotShow=save, show=1, fill=1, ExpErrors=1,
                 StandardError=1, logplot=0,
                 HistMin='auto', HistMax='auto',
                 PlotOnly=[0,3,5,7], titles='off', plt_mean_yaxis=[0.2,0.42])

#

#

obj_a = analysis(_con2dds_d, save_loc='Paper/graphs/_con2dds_d_all', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_basic(sel_dict='na', Save_NotShow=save, show=1, fill=1, ExpErrors=1,
                 StandardError=1, logplot=0,
                 HistMin=0, HistMax='auto',
                 PlotOnly='all', titles='off')  # HistBins=0.02, HistMin=0, HistMax='auto' PlotOnly=[3+5,4+5]

obj_a = analysis(_con2dds_d, save_loc='Paper/graphs/_con2dds_d_main', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_basic(sel_dict='na', Save_NotShow=save, show=1, fill=1, ExpErrors=1,
                 StandardError=1, logplot=0,
                 HistMin='auto', HistMax='auto',
                 PlotOnly=[0,1,2,4,6], titles='off', plt_mean_yaxis=[0.2,0.42])  #

obj_a = analysis(_con2dds_d, save_loc='Paper/graphs/_con2dds_d_sup', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_basic(sel_dict='na', Save_NotShow=save, show=1, fill=1, ExpErrors=1,
                 StandardError=1, logplot=0,
                 HistMin='auto', HistMax='auto',
                 PlotOnly=[0,3,5,7], titles='off', plt_mean_yaxis=[0.2,0.42])

#

#

#

#

# # MG plots
save = 1
form = 'png'

obj_a = analysis(RRN_bestG, save_loc='Paper/MG/bestG/RRN', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_mg(Save_NotShow=save, Bgeno=1, Dgeno=0, VC=0, VP=0, VoW=0, ViW=0,
             VoW_ani=0, ViW_ani=0, VoiW_ani=0, VloW_ani=0, VliW_ani=0, Specific_Cir_Rep=[1,0],
             titles='off')
obj_a = analysis(DRN_bestG, save_loc='Paper/MG/bestG/DRN', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_mg(Save_NotShow=save, Bgeno=1, Dgeno=0, VC=0, VP=0, VoW=0, ViW=0,
             VoW_ani=0, ViW_ani=0, VoiW_ani=0, VloW_ani=0, VliW_ani=0, Specific_Cir_Rep=[0,1],
             titles='off')

#

obj_a = analysis(RRN_config, save_loc='Paper/MG/config/RRN', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_mg(Save_NotShow=save, Bgeno=0, Dgeno=0, VC=1, VP=0, VoW=0, ViW=0,
             VoW_ani=0, ViW_ani=0, VoiW_ani=0, VloW_ani=0, VliW_ani=0, Specific_Cir_Rep=[1,0],
             titles='off')
obj_a = analysis(DRN_config, save_loc='Paper/MG/config/DRN', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_mg(Save_NotShow=save, Bgeno=0, Dgeno=0, VC=1, VP=0, VoW=0, ViW=0,
             VoW_ani=0, ViW_ani=0, VoiW_ani=0, VloW_ani=0, VliW_ani=0, Specific_Cir_Rep=[0,0],
             titles='off')

#

obj_a = analysis(RRN_shuffle, save_loc='Paper/MG/shuffle/RRN', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_mg(Save_NotShow=save, Bgeno=0, Dgeno=0, VC=0, VP=1, VoW=0, ViW=0,
             VoW_ani=0, ViW_ani=0, VoiW_ani=0, VloW_ani=0, VliW_ani=0, Specific_Cir_Rep=[0,0],
             titles='off')
obj_a = analysis(DRN_shuffle, save_loc='Paper/MG/shuffle/DRN', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_mg(Save_NotShow=save, Bgeno=0, Dgeno=0, VC=0, VP=1, VoW=0, ViW=0,
             VoW_ani=0, ViW_ani=0, VoiW_ani=0, VloW_ani=0, VliW_ani=0, Specific_Cir_Rep=[4,0],
             titles='off')









#

# fin
