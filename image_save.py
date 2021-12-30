# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 15:30:59 2021

@author: 邵键帆
"""


"""
This program is aimed at some images output mainpulation
"""

import numpy as np
import pandas as pd


##################################################### image to excel ######################################


def image2excel(image,excel_name):
    df_list=[]
    for i in range(image.shape[2]):
        df=pd.DataFrame(image[:,:,i])
        df_list.append(df)
    with pd.ExcelWriter("{}.xlsx".format(excel_name)) as w:
        for i in range(image.shape[2]):
            df_list[i].to_excel(w,sheet_name="channel{}".format(i+1),index=False)
    




##################################################### image to excel ######################################
