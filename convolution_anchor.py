# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 15:24:52 2021

@author: 邵键帆
"""


"""
This program demonstrate the more general convolution
the convolution filter with the random anchor(not the center of the filter)
"""

#import the relevant package
import numpy as np

from convolution import show_with_matplotlib_array,bgr_imread


#define a class to execute the convolution transformation with specific 
#anchor center
class convolution_anchor_trans():
    
    #initialize the relevant parameters
    def __init__(self,array,kernel,s,t,anchor_row,anchor_col):
        #kernel: kernel used in the convolutional process
        #array: the array waiting for convolutional transformation
        #s: sliding steps of the row(vertical dimension)
        #t: sliding steps of the column(horizontal dimension)
        #anchor_row: row of the anchor center
        #anchor_col: column of the anchor center
        self.kernel=kernel
        self.array=array
        self.s=s
        self.t=t
        self.anchor_row=anchor_row
        self.anchor_col=anchor_col
        
    #return the padding num of the row and column dimension
    def pad_num(self):
        m,n=self.array.shape[0],self.array.shape[1]
        k,j=self.kernel.shape[0],self.kernel.shape[1]
        up_kernel_shape=(self.anchor_row*2)+1
        down_kernel_shape=((k-self.anchor_row-1)*2)+1
        left_kernel_shape=(self.anchor_col*2)+1
        right_kernel_shape=((j-self.anchor_col-1)*2)+1
        p_up,p_down=((self.s-1)*m-self.s+up_kernel_shape)/2,((self.s-1)*m-self.s+down_kernel_shape)/2
        q_left,q_right=((self.t-1)*n-self.t+left_kernel_shape)/2,((self.t-1)*n-self.t+right_kernel_shape)/2
        
        return (int(p_up),int(p_down)),(int(q_left),int(q_right))

    #return the array after padding transformation
    def pad_zero(self,row_num=0,column_num=0):
        row,col=self.pad_num()
        padding_array=np.pad(self.array,(row,col),"constant",constant_values=(row_num,column_num))
        return padding_array


############################################################################################## TEST ##############################################################################

"""

lenna_array=bgr_imread("pics\\lenna.png")[:,:,0]
filter_1=np.arange(1,17).reshape(4,4)
lenna_array_convolution_anchor=convolution_anchor_trans(lenna_array,filter_1,1,1,1,1)
lenna_padding_array=lenna_array_convolution_anchor.pad_zero()
show_with_matplotlib_array(lenna_padding_array,"lenna_padding_array")

"""
############################################################################################## TEST ##############################################################################


