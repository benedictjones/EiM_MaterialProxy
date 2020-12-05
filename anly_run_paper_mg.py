from mod_analysis.Analysis import analysis




######################################
# # Paper data locations


# # 2DDS
RRN = 'Results/2020_12_01/__13_06_42__2DDS__R_RN__EiM'
DRN = 'Results/2020_12_01/__14_01_22__2DDS__D_RN__EiM'

# # con2DDS
cDRN = 'Results/2020_12_01/__13_15_41__con2DDS__D_RN__EiM'
cNLRN = 'Results/2020_12_01/__13_08_19__con2DDS__NL_RN__EiM'



# ###########################################
# # Generate plots for paper
# ###########################################


# # Fitness Plots
save = 1
form = 'png'  # or 'pdf' or 'png'



#

"""
obj_a = analysis(RRN, save_loc='Paper/mg/best/RRN', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_mg(Save_NotShow=save, Bgeno=1, Dgeno=0, VC=0, VP=0, VoW=0, ViW=0,
             #
             titles='off',
             fig_letter='a',
             figsize=[2.7,2.6],
             #
             Specific_Cir_Rep=[0,0]  # select a specific [circuit, repetition] to plot
             )

obj_a = analysis(DRN, save_loc='Paper/mg/best/DRN', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_mg(Save_NotShow=save, Bgeno=1, Dgeno=0, VC=0, VP=0, VoW=0, ViW=0,
             #
             titles='off',
             fig_letter='b',
             figsize=[2.7,2.6],
             #
             Specific_Cir_Rep=[3,0]  # select a specific [circuit, repetition] to plot
             )

obj_a = analysis(cDRN, save_loc='Paper/mg/best/cDRN', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_mg(Save_NotShow=save, Bgeno=1, Dgeno=0, VC=0, VP=0, VoW=0, ViW=0,
             #
             titles='off',
             fig_letter='c',
             figsize=[2.7,2.6],
             #
             Specific_Cir_Rep=[3,0]  # select a specific [circuit, repetition] to plot
             )

obj_a = analysis(cNLRN, save_loc='Paper/mg/best/cNLRN', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_mg(Save_NotShow=save, Bgeno=1, Dgeno=0, VC=0, VP=0, VoW=0, ViW=0,
             #
             titles='off',
             fig_letter='d',
             figsize=[2.7,2.6],
             #
             Specific_Cir_Rep=[0,0]  # select a specific [circuit, repetition] to plot
             )
#"""

#

#

#
"""
fs = [3.1,2.9]
fs = [2.2,2.1]
#fs = 'na'
obj_a = analysis(RRN, save_loc='Paper/mg/Vconfig/RRN', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_mg(Save_NotShow=save, Bgeno=0, Dgeno=0, VC=1, VP=0, VoW=0, ViW=0,
             #
             titles='off',
             fig_letter='a',
             figsize=fs,
             #
             Specific_Cir_Rep=[3,0]  # select a specific [circuit, repetition] to plot
             )

obj_a = analysis(cDRN, save_loc='Paper/mg/Vconfig/DRN', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_mg(Save_NotShow=save, Bgeno=0, Dgeno=0, VC=1, VP=0, VoW=0, ViW=0,
             #
             titles='off',
             fig_letter='c',
             figsize=fs,
             #
             Specific_Cir_Rep=[0,0]  # select a specific [circuit, repetition] to plot
             )

obj_a = analysis(cNLRN, save_loc='Paper/mg/Vconfig/cNLRN', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_mg(Save_NotShow=save, Bgeno=0, Dgeno=0, VC=1, VP=0, VoW=0, ViW=0,
             #
             titles='off',
             fig_letter='b',
             figsize=fs,
             #
             Specific_Cir_Rep=[1,0]  # select a specific [circuit, repetition] to plot
             )
#"""

#

#

#
#"""
fs = [2.8,2.7] # prev
fs = [2.2,2.1]
#fs = 'na'
obj_a = analysis(RRN, save_loc='Paper/mg/Perm/RRN', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_mg(Save_NotShow=save, Bgeno=0, Dgeno=0, VC=0, VP=1, VoW=0, ViW=0,
             #
             titles='off',
             fig_letter='a',
             figsize=fs,
             #
             Specific_Cir_Rep=[1,0]  # select a specific [circuit, repetition] to plot
             )

obj_a = analysis(cDRN, save_loc='Paper/mg/Perm/DRN', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_mg(Save_NotShow=save, Bgeno=0, Dgeno=0, VC=0, VP=1, VoW=0, ViW=0,
             #
             titles='off',
             fig_letter='c',
             figsize=fs,
             #
             Specific_Cir_Rep=[1,0]  # select a specific [circuit, repetition] to plot
             )

obj_a = analysis(cNLRN, save_loc='Paper/mg/Perm/cNLRN', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_mg(Save_NotShow=save, Bgeno=0, Dgeno=0, VC=0, VP=1, VoW=0, ViW=0,
             #
             titles='off',
             fig_letter='b',
             figsize=fs,
             #
             Specific_Cir_Rep=[2,0]  # select a specific [circuit, repetition] to plot
             )
#"""

#

#

#

#

"""
obj_a = analysis(RRN, save_loc='Paper/mg/Weights/RRN', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_mg(Save_NotShow=save, Bgeno=0, Dgeno=0, VC=0, VP=0, VoW=1, ViW=1,
             #
             titles='off',
             fig_letter='a',
             figsize=[3.5,3.4],
             #
             Specific_Cir_Rep=[0,0]  # select a specific [circuit, repetition] to plot
             )

obj_a = analysis(cDRN, save_loc='Paper/mg/Weights/DRN', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_mg(Save_NotShow=save, Bgeno=0, Dgeno=0, VC=0, VP=0, VoW=1, ViW=1,
             #
             titles='off',
             fig_letter='b',
             figsize=[3.5,3.4],
             #
             Specific_Cir_Rep=[1,0]  # select a specific [circuit, repetition] to plot
             )

obj_a = analysis(cNLRN, save_loc='Paper/mg/Weights/cNLRN', format=form)  # dir_NameList=FilePAram,
obj_a.Plt_mg(Save_NotShow=save, Bgeno=0, Dgeno=0, VC=0, VP=0, VoW=1, ViW=1,
             #
             titles='off',
             fig_letter='c',
             figsize=[3.5,3.4],
             #
             Specific_Cir_Rep=[2,0]  # select a specific [circuit, repetition] to plot
             )
#"""

#

# fin
