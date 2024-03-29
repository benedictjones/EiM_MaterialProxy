import numpy as np
import pandas as pd
import os


def Load_NewWeighted_SaveTemp(DataDir, sys, rep, TestVerify):

    save_loc = "%s/NewData/DataWithWeigthAttribute.h5" % (DataDir)

    # # make temp save dir
    dir = "Module_LoadData/TempData"
    if not os.path.exists(dir):
        os.makedirs(dir)

    # # save to a temp location
    if TestVerify == 0:
        data = pd.read_hdf(save_loc, 'all')
        save_loc1 = "%s/TempData.h5" % (dir)
        data.to_hdf(save_loc1, key="%cird_rep%d/all" % (sys, rep), mode='a')

    elif TestVerify == 1:

        data = pd.read_hdf(save_loc, 'training')
        save_loc1 = "%s/TempData.h5" % (dir)
        data.to_hdf(save_loc1, key="%cird_rep%d/training" % (sys, rep), mode='a')

        data = pd.read_hdf(save_loc, 'veri')
        save_loc1 = "%s/TempData.h5" % (dir)
        data.to_hdf(save_loc1, key="%cird_rep%d/veri" % (sys, rep), mode='a')

    return
