# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 16:04:08 2021

@author: 邵键帆
"""


"""
This program aimed at morphological transformation(for the black white pic: 0/255 for inner point set of bgr set for margin point)
all of the picture are imported in (0,256)-form, no need to transform to (0,1)-form
this convolution methodology refers to the topology transformation
"""


#import the relevant package

import numpy as np
from convolution import show_with_matplotlib,show_with_matplotlib_array,bgr_imread,image_merge
from convolution_anchor import convolution_anchor_trans
from arithmetic_operation import array_binary_threshold,image_binary_threshold,saturation_array,saturation_image
import pandas as pd


##################################################################### Morphological Filter Reference #####################################################

rectangular_filter=np.array([[1,1,1,1,1],
                             [1,1,1,1,1],
                             [1,1,1,1,1],
                             [1,1,1,1,1],
                             [1,1,1,1,1]],dtype="uint8")

elliptical_filter=np.array([[0,0,1,0,0],
                            [1,1,1,1,1],
                            [1,1,1,1,1],
                            [1,1,1,1,1],
                            [0,0,1,0,0]],dtype="uint8")

cross_filter=np.array([[0,0,1,0,0],
                       [0,0,1,0,0],
                       [1,1,1,1,1],
                       [0,0,1,0,0],
                       [0,0,1,0,0]],dtype="uint8")




##################################################################### Morphological Filter Reference #####################################################



##################################################################### Morphological transformation #####################################################

#define a class to execute the morphological transformation
class array_morpho_trans():
    
    #initialize the relevant parameters for the array_morpho_trans class
    def __init__(self,array,kernel,anchor_row,anchor_col,s=1,t=1):
        
        self.array=array
        self.kernel=kernel
        self.anchor_row=anchor_row
        self.anchor_col=anchor_col
        self.s=s
        self.t=t
    
    #return an array which is eroded
    def array_erode(self):
        invert_kernel=np.invert(self.kernel)+2
        kernel_one=np.ones(self.kernel.shape)
        padding_trans=convolution_anchor_trans(self.array,self.kernel,self.s,self.t,self.anchor_row,self.anchor_col)
        padding_array=padding_trans.pad_zero()
        copy_padding_array=padding_array.copy()
        padding_row,padding_col=padding_trans.pad_num()
        (padding_row_start,padding_row_end)=padding_row
        (padding_col_start,padding_col_end)=padding_col
        
        for i in range(self.anchor_row,padding_array.shape[0]-(self.kernel.shape[0]-self.anchor_row-1)):
            for j in range(self.anchor_col,padding_array.shape[1]-(self.kernel.shape[1]-self.anchor_col-1)):
                
                
                if padding_array[i,j] != 0:
                    structure_list=[]
                    for p in range(i-self.anchor_row,i+(self.kernel.shape[0]-self.anchor_row)):
                        structure_line_list=[]
                        for q in range(j-self.anchor_col,j+(self.kernel.shape[1]-self.anchor_col)):
                            
                            structure_line_list.append(padding_array[p,q])
                        structure_list.append(structure_line_list)
                    structure_array=np.array(structure_list).astype(int)
                    binary_structure_array=array_binary_threshold(structure_array)
                    
                    judge_array=(self.kernel & binary_structure_array)|(invert_kernel)
                    
                    if not (judge_array==kernel_one).all():
                        copy_padding_array[i,j]=0
        
        return copy_padding_array[padding_row_start:-padding_row_end,padding_col_start:-padding_col_end]


    #return an array which is dilated
    def array_dilate(self):
        invert_kernel=np.invert(self.kernel)+2
        kernel_one=np.ones(self.kernel.shape)
        padding_trans=convolution_anchor_trans(self.array,self.kernel,self.s,self.t,self.anchor_row,self.anchor_col)
        padding_array=padding_trans.pad_zero()
        copy_padding_array=padding_array.copy()
        padding_row,padding_col=padding_trans.pad_num()
        (padding_row_start,padding_row_end)=padding_row
        (padding_col_start,padding_col_end)=padding_col
        
        for i in range(self.anchor_row,padding_array.shape[0]-(self.kernel.shape[0]-self.anchor_row-1)):
            for j in range(self.anchor_col,padding_array.shape[1]-(self.kernel.shape[1]-self.anchor_col-1)):
                
                
                if padding_array[i,j] != 0:
                    structure_list=[]
                    for p in range(i-self.anchor_row,i+(self.kernel.shape[0]-self.anchor_row)):
                        structure_line_list=[]
                        for q in range(j-self.anchor_col,j+(self.kernel.shape[1]-self.anchor_col)):
                            
                            structure_line_list.append(padding_array[p,q])
                        structure_list.append(structure_line_list)
                    structure_array=np.array(structure_list).astype(int)
                    binary_structure_array=array_binary_threshold(structure_array)
                    
                    judge_array=binary_structure_array | invert_kernel
                    
                    for m in range(judge_array.shape[0]):
                        for n in range(judge_array.shape[1]):
                            if judge_array[m,n] == 0:
                                copy_padding_array[i+(m-self.anchor_row),j+(n-self.anchor_col)]=255
                    
        
        return padding_array[padding_row_start:-padding_row_end,padding_col_start:-padding_col_end]

    #retrun an array transformed by opening operation
    def array_opening(self):
        
        erode_array=self.array_erode()
        
        kernel_symmetry_judge=(self.kernel==np.transpose(self.kernel)).all()
        if not kernel_symmetry_judge:
            dilate_kernel=np.invert(self.kernel)+2
        else:
            dilate_kernel=self.kernel
        dilate_array=self.array_dilate()
        
        return dilate_array
        
    #return an array transformed by closing operation
    def array_closing(self):
        
        kernel_symmetry_judge=(self.kernel==np.transpose(self.kernel)).all()
        if not kernel_symmetry_judge:
            dilate_kernel=np.invert(self.kernel)+2
        else:
            dilate_kernel=self.kernel
        dilate_array=self.array_dilate()  
        
        erode_array=self.array_erode()
        
        return erode_array
    
    
    def array_gradient(self):
        gradient_array=abs(self.array_dialte()-self.array_erode())
        return gradient_array
    
    
    def array_white_tophat(self):
        tophat_array=self.array-self.array_opening()
        return tophat_array
    
    def array_black_tophat(self):
        tophat_array=self.array_closing()-self.array
        return tophat_array






#define an class to execute the morphological transformation on image(3 channels array)
class image_morpho_trans():
    
    def __init__(self,image,kernel,anchor_row,anchor_col,s=1,t=1):
        self.image=image
        self.kernel=kernel
        self.anchor_row=anchor_row
        self.anchor_col=anchor_col
        self.s=s
        self.t=t


    #return an image which is eroded
    def image_erode(self):
        channel_list=[]
        for channel in range(self.image.shape[2]):
            array=self.image[:,:,channel]
            invert_kernel=np.invert(self.kernel)+2
            kernel_one=np.ones(self.kernel.shape)
            padding_trans=convolution_anchor_trans(array,self.kernel,self.s,self.t,self.anchor_row,self.anchor_col)
            padding_array=padding_trans.pad_zero()
            copy_padding_array=padding_array.copy()
            padding_row,padding_col=padding_trans.pad_num()
            (padding_row_start,padding_row_end)=padding_row
            (padding_col_start,padding_col_end)=padding_col
            for i in range(self.anchor_row,padding_array.shape[0]-(self.kernel.shape[0]-self.anchor_row-1)):
                for j in range(self.anchor_col,padding_array.shape[1]-(self.kernel.shape[1]-self.anchor_col-1)):
                    
                    
                    if padding_array[i,j] != 0:
                        structure_list=[]
                        for p in range(i-self.anchor_row,i+(self.kernel.shape[0]-self.anchor_row)):
                            
                            structure_line_list=[]
                            for q in range(j-self.anchor_col,j+(self.kernel.shape[1]-self.anchor_col)):
                                
                                structure_line_list.append(padding_array[p,q])
                            structure_list.append(structure_line_list)
                        structure_array=np.array(structure_list).astype(int)
                        binary_structure_array=array_binary_threshold(structure_array)
                        
                        judge_array=(self.kernel & binary_structure_array)|(invert_kernel)
                        if not (judge_array==kernel_one).all():
                            copy_padding_array[i,j]=0
                            
            channel_list.append(copy_padding_array[padding_row_start:-padding_row_end,padding_col_start:-padding_col_end])
        
        trans_image=image_merge(channel_list).astype(int)
        
        return trans_image
    
    #return an image which is dilated
    def image_dilate(self):
        channel_list=[]
        for channel in range(self.image.shape[2]):
            array=self.image[:,:,channel]
            invert_kernel=np.invert(self.kernel)+2
            kernel_one=np.ones(self.kernel.shape)
            padding_trans=convolution_anchor_trans(array,self.kernel,self.s,self.t,self.anchor_row,self.anchor_col)
            padding_array=padding_trans.pad_zero()
            copy_padding_array=padding_array.copy()
            padding_row,padding_col=padding_trans.pad_num()
            (padding_row_start,padding_row_end)=padding_row
            (padding_col_start,padding_col_end)=padding_col
            for i in range(self.anchor_row,padding_array.shape[0]-(self.kernel.shape[0]-self.anchor_row-1)):
                for j in range(self.anchor_col,padding_array.shape[1]-(self.kernel.shape[1]-self.anchor_col-1)):
                    
                    
                    if padding_array[i,j] != 0:
                        structure_list=[]
                        for p in range(i-self.anchor_row,i+(self.kernel.shape[0]-self.anchor_row)):
                            
                            structure_line_list=[]
                            for q in range(j-self.anchor_col,j+(self.kernel.shape[1]-self.anchor_col)):
                                
                                structure_line_list.append(padding_array[p,q])
                            structure_list.append(structure_line_list)
                        structure_array=np.array(structure_list).astype(int)
                        binary_structure_array=array_binary_threshold(structure_array)
                        
                        judge_array=binary_structure_array | invert_kernel
                        for m in range(judge_array.shape[0]):
                            for n in range(judge_array.shape[1]):
                                if judge_array[m,n] == 0:
                                    copy_padding_array[i+(m-self.anchor_row),j+(n-self.anchor_col)]=255
                            
            channel_list.append(copy_padding_array[padding_row_start:-padding_row_end,padding_col_start:-padding_col_end])
        
        trans_image=image_merge(channel_list).astype(int)
        
        return trans_image

    #return an image transformed by opening operation
    def image_opening(self):
        
        erode_image=self.image_erode()
        
        kernel_symmetry_judge=(self.kernel==np.transpose(self.kernel)).all()
        if not kernel_symmetry_judge:
            dilate_kernel=np.invert(self.kernel)+2
        else:
            dilate_kernel=self.kernel
        dilate_image=self.image_dilate()
        
        return dilate_image
        
    #return an image transformed by closing operation
    def image_closing(self):
        
        kernel_symmetry_judge=(self.kernel==np.transpose(self.kernel)).all()
        if not kernel_symmetry_judge:
            dilate_kernel=np.invert(self.kernel)+2
        else:
            dilate_kernel=self.kernel
        dilate_image=self.image_dilate()  
        
        erode_image=self.image_erode()
        
        return erode_image
    
    def image_gradient(self):
        gradient_image=abs((self.image_dilate()-self.image_erode()))
        return gradient_image
    
    def image_white_tophat(self):
        tophat_image=-(self.image-self.image_opening())
        
        
        
        
        return tophat_image
    
    def image_black_tophat(self):
        tophat_image=-(self.image_closing()-self.image)
        
        return tophat_image
    
    
                

    


##################################################################### Morphological transformation #####################################################


############################################################################################## TEST ##############################################################################


"""
                
test_kernel=np.array([[1,1,1,1,1],
                      [1,1,1,1,1],
                      [1,1,1,1,0],
                      [1,1,1,1,1],
                      [1,1,1,1,1]],dtype="uint8")


test_image=bgr_imread("pics\\test1.png")
test_image_trans=image_morpho_trans(test_image,test_kernel,2,2)


test_image_erode=test_image_trans.image_erode()
test_image_dilate=test_image_trans.image_dilate()
test_image_opening=test_image_trans.image_opening()
test_image_closing=test_image_trans.image_closing()
test_image_gradient=test_image_trans.image_gradient()
test_image_white_tophat=test_image_trans.image_white_tophat()
test_image_black_tophat=test_image_trans.image_black_tophat()




show_with_matplotlib(test_image,"test_image_original")
show_with_matplotlib(test_image_erode,"test_image_erode")
show_with_matplotlib(test_image_dilate,"test_image_dilate")
show_with_matplotlib(test_image_opening,"test_image_opening")
show_with_matplotlib(test_image_closing,"test_image_closing")
show_with_matplotlib(test_image_gradient,"test_image_gradient")
show_with_matplotlib(test_image_white_tophat,"test_image_white_tophat")
show_with_matplotlib(test_image_black_tophat,"test_image_black_tophat")

"""



"""

#the morphological transformation is unmature for one channel image(array)

test_kernel=np.array([[0,0,1,0,0],
                      [0,0,1,0,0],
                      [1,1,1,1,1],
                      [0,0,1,0,0],
                      [0,0,1,0,0]],dtype="uint8")




test_array=bgr_imread("pics\\test1.png")[:,:,0]
test_array_trans=array_morpho_trans(test_array,test_kernel,2,2)
test_array_erode=test_array_trans.array_erode()
test_array_dilate=test_array_trans.array_dilate()
test_array_opening=test_array_trans.array_opening()
test_array_closing=test_array_trans.array_closing()
test_array_white_tophat=test_array_trans.array_white_tophat()
test_array_black_tophat=test_array_trans.array_black_tophat()


show_with_matplotlib_array(test_array,"test_array_original")
show_with_matplotlib_array(test_array_erode,"test_array_erode")
show_with_matplotlib_array(test_array_dilate,"test_array_dilate")
show_with_matplotlib_array(test_array_opening,"test_array_opening")
show_with_matplotlib_array(test_array_closing,"test_array_closing")
show_with_matplotlib_array(test_array_white_tophat,"test_array_white_tophat")
show_with_matplotlib_array(test_array_black_tophat,"test_array_black_tophat")


"""


############################################################################################## TEST ##############################################################################


