# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 11:31:08 2021

@author: 邵键帆
"""



"""
This program is aimed at image resize
The underlying algorithm is interpolation
"""




#import relevant package

import numpy as np
from convolution import show_with_matplotlib_array,show_with_matplotlib,bgr_imread,image_split,image_merge
import math
from convolution_anchor import convolution_anchor_trans





#define the function to execute the saturation transformation for one channel array
#Because the relvant visualization tool limit the color range to \
#[0,255], so we need to converse the out-bound number to the limited 
#range    
def saturation_array(array,up_threshold_value=255,low_threshold_value=0):
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i,j] > up_threshold_value:
                array[i,j] = 255
            elif array[i,j] < low_threshold_value:
                array[i,j] = 0
            
    return array


#define the function to execute the saturation transformation for image 
#Because the relvant visualization tool limit the color range to \
#[0,255], so we need to converse the out-bound number to the limited 
#range    
def saturation_image(image,up_threshold_value=255,low_threshold_value=0):
    b,g,r=image[:,:,0],image[:,:,1],image[:,:,2]
    b_saturation_array=saturation_array(b,up_threshold_value,low_threshold_value)
    g_saturation_array=saturation_array(g,up_threshold_value,low_threshold_value)
    r_saturation_array=saturation_array(r,up_threshold_value,low_threshold_value)
    saturation_trans_image=image_merge([b_saturation_array,g_saturation_array,r_saturation_array])
    return saturation_trans_image





################################################################################### normal interpolation ###########################################################################

#return the cubic function result
def bi_cubic_function(x,a=-0.5):
    if abs(x) <= 1:
        value = (a+2)*((abs(x))**3)-(a+3)*((abs(x))**2)+1
    elif abs(x) < 2 and abs(x) > 1:
        value = a*((abs(x))**3)-5*a*((abs(x))**2)+8*a*(abs(x))-4*a
    else:
        value = 0
    
    return value


#define a class to execute interpolation (image resize)
class array_inter_trans:
    
    #initialize the relevant parameters
    def __init__(self,array,trans_row,trans_col):
        #array: the array waiting for transformation
        #trans_row: row of the target image
        #trans_col: column of the target image 
        self.array=array
        self.trans_row=trans_row
        self.trans_col=trans_col
        
    
    #return the array(one channel image) transformed by nearest neighborhood interpolation
    def array_inter_nn(self):
        (original_row,original_col) = self.array.shape
        trans_list=[]
        for i in range(self.trans_row):
            line_list=[]
            for j in range(self.trans_col):
                
                trans_value=self.array[round(((original_row-1)/(self.trans_row-1))*i),round(((original_col-1)/(self.trans_col-1))*j)]
                line_list.append(trans_value)
            
            trans_list.append(line_list)
        return np.array(trans_list).astype(int)


    #return the array(one channel image) transformed by bi-linear interpolation
    def array_inter_bl(self):
        (original_row,original_col) = self.array.shape
        trans_list=[]
        for i in range(self.trans_row):
            line_list=[]
            for j in range(self.trans_col):
                float_i=((original_row-1)/(self.trans_row-1))*i
                float_j=((original_col-1)/(self.trans_col-1))*j
                i_ceil,i_floor=math.ceil(float_i),math.floor(float_i)
                j_ceil,j_floor=math.ceil(float_j),math.floor(float_j)
                u=float_i-i_floor
                v=float_j-j_floor
                trans_value=(1-u)*(1-v)*self.array[i_floor,j_floor]+u*(1-v)*self.array[i_ceil,j_floor]+(1-u)*v*self.array[i_floor,j_ceil]+\
                    u*v*self.array[i_ceil,j_ceil]
                line_list.append(trans_value)
            trans_list.append(line_list)
        
        return np.array(trans_list).astype(int)


    #return the array transformed by bi-cubic (bi refer to horizontal and vertical dimension) interpolation
    def array_inter_bicubic(self,s=1,t=1,anchor_row=1,anchor_col=1):
        fake_kernel=np.ones((4,4),dtype="uint8")
        convolution_anchor_trans_class=convolution_anchor_trans(self.array,fake_kernel,s,t,anchor_row,anchor_col)
        padding_array=convolution_anchor_trans_class.pad_zero()
        (original_row,original_col) = self.array.shape    
        trans_list=[]
        for i in range(self.trans_row):
            line_list=[]
            for j in range(self.trans_col):
                float_i=((original_row-1)/(self.trans_row-1))*i
                float_j=((original_col-1)/(self.trans_col-1))*j
                i_floor,j_floor=math.floor(float_i),math.floor(float_j)
                padding_i_floor,padding_j_floor=i_floor+anchor_row,j_floor+anchor_col
                padding_float_i,padding_float_j=float_i+anchor_row,float_j+anchor_col
                
                pixel_list,weight_list=[],[]
                for p in range(padding_i_floor-anchor_row,padding_i_floor+(fake_kernel.shape[0]-anchor_row)):
                    pixel_line,weight_line=[],[]
                    for q in range(padding_j_floor-anchor_row,padding_j_floor+(fake_kernel.shape[1]-anchor_col)):
                        pixel_value=padding_array[p,q]
                        weight_value=bi_cubic_function(p-padding_float_i)*bi_cubic_function(q-padding_float_j)
                        
                        pixel_line.append(pixel_value)
                        weight_line.append(weight_value)
                    
                    pixel_list.append(pixel_line)
                    weight_list.append(weight_line)
                
                pixel_array=np.array(pixel_list)
                weight_array=np.array(weight_list)
                
                trans_value=np.sum(np.multiply(pixel_array,weight_array))
            
                line_list.append(trans_value)
            
            trans_list.append(line_list)
        
        trans_array=np.array(trans_list).astype(int)
        saturation_trans_array=saturation_array(trans_array)
        
        return saturation_trans_array





#define a class to execute the interpolation to the image(3 channels array)

class image_inter_trans():
    #initialize the relevant parameters 
    def __init__(self,image,trans_row,trans_col):
        #image:the image waiting for interpolation transformation
        #trans_row: the rows of the target image 
        #trans_col: the columns of the target image
        self.image=image
        self.trans_row=trans_row
        self.trans_col=trans_col


    #return the image transformed by the bi-linear interpolation
    def image_inter_bl(self):
        (original_row,original_col,original_channel) = self.image.shape
        channel_list=[]
        for channel in range(original_channel):
            array=self.image[:,:,channel]
            trans_list=[]
            for i in range(self.trans_row):
                line_list=[]
                for j in range(self.trans_col):
                    float_i=((original_row-1)/(self.trans_row-1))*i
                    float_j=((original_col-1)/(self.trans_col-1))*j
                    i_ceil,i_floor=math.ceil(float_i),math.floor(float_i)
                    j_ceil,j_floor=math.ceil(float_j),math.floor(float_j)
                    u=float_i-i_floor
                    v=float_j-j_floor
                    trans_value=(1-u)*(1-v)*array[i_floor,j_floor]+u*(1-v)*array[i_ceil,j_floor]+(1-u)*v*array[i_floor,j_ceil]+\
                        u*v*array[i_ceil,j_ceil]
                    line_list.append(trans_value)
                trans_list.append(line_list)
            trans_array=np.array(trans_list).astype(int)
            channel_list.append(trans_array)
        
        trans_image=image_merge(channel_list)
        return trans_image




    #return the image transformed by the nearest neightborhood interpolation
    def image_inter_nn(self):
        (original_row,original_col,original_channel) = self.image.shape
        channel_list=[]
        for channel in range(original_channel):
            array=self.image[:,:,channel]
            trans_list=[]
            for i in range(self.trans_row):
                line_list=[]
                for j in range(self.trans_col):
                    
                    trans_value=array[round(((original_row-1)/(self.trans_row-1))*i),round(((original_col-1)/(self.trans_col-1))*j)]
                    line_list.append(trans_value)
                
                trans_list.append(line_list)
            trans_array=np.array(trans_list).astype(int)
            channel_list.append(trans_array)
        trans_image=image_merge(channel_list)
        
        return trans_image




    #return the image transformed by the bi-cubic interpolation
    def image_inter_bicubic(self,s=1,t=1,anchor_row=1,anchor_col=1):
        (original_row,original_col,original_channel)=self.image.shape
        channel_list=[]
        
        for channel in range(original_channel):
            array=self.image[:,:,channel]
            fake_kernel=np.ones((4,4),dtype="uint8")
            convolution_anchor_trans_class=convolution_anchor_trans(array,fake_kernel,s,t,anchor_row,anchor_col)
            padding_array=convolution_anchor_trans_class.pad_zero()
            (original_row,original_col) = array.shape    
            trans_list=[]
            for i in range(self.trans_row):
                line_list=[]
                for j in range(self.trans_col):
                    float_i=((original_row-1)/(self.trans_row-1))*i
                    float_j=((original_col-1)/(self.trans_col-1))*j
                    i_floor,j_floor=math.floor(float_i),math.floor(float_j)
                    padding_i_floor,padding_j_floor=i_floor+anchor_row,j_floor+anchor_col
                    padding_float_i,padding_float_j=float_i+anchor_row,float_j+anchor_col
                    
                    pixel_list,weight_list=[],[]
                    for p in range(padding_i_floor-anchor_row,padding_i_floor+(fake_kernel.shape[0]-anchor_row)):
                        pixel_line,weight_line=[],[]
                        for q in range(padding_j_floor-anchor_row,padding_j_floor+(fake_kernel.shape[1]-anchor_col)):
                            pixel_value=padding_array[p,q]
                            weight_value=bi_cubic_function(p-padding_float_i)*bi_cubic_function(q-padding_float_j)
                            
                            pixel_line.append(pixel_value)
                            weight_line.append(weight_value)
                        
                        pixel_list.append(pixel_line)
                        weight_list.append(weight_line)
                    
                    pixel_array=np.array(pixel_list)
                    weight_array=np.array(weight_list)
                    
                    trans_value=np.sum(np.multiply(pixel_array,weight_array))
                
                    line_list.append(trans_value)
                
                trans_list.append(line_list)
            
            trans_array=np.array(trans_list)
            channel_list.append(trans_array)
        
        trans_image=image_merge(channel_list).astype(int)
        saturation_trans_image=saturation_image(trans_image)
        return trans_image



################################################################################### normal interpolation ###########################################################################


############################################################################################## TEST ##############################################################################

"""

lenna_array=bgr_imread("pics\\lenna.png")[:,:,0]
lenna_inter_trans=array_inter_trans(lenna_array,300,300)
lenna_nn_resize_array=lenna_inter_trans.array_inter_nn()
lenna_bl_resize_array=lenna_inter_trans.array_inter_bl()
lenna_bicubic_resize_array=lenna_inter_trans.array_inter_bicubic()
show_with_matplotlib_array(lenna_nn_resize_array,"lenna_nn_resize_array")
show_with_matplotlib_array(lenna_bl_resize_array,"lenna_bl_resize_array")
show_with_matplotlib_array(lenna_bicubic_resize_array,"lenna_bicubic_resize_array")

"""

"""

lenna_image=bgr_imread("pics\\lenna.png")
lenna_inter_trans=image_inter_trans(lenna_image,500,500)
lenna_nn_resize_image=lenna_inter_trans.image_inter_nn()
lenna_bl_resize_image=lenna_inter_trans.image_inter_bl()
lenna_bicubic_resize_image=lenna_inter_trans.image_inter_bicubic()
show_with_matplotlib(lenna_nn_resize_image,"lenna_nn_resize_image")
show_with_matplotlib(lenna_bl_resize_image,"lenna_bl_resize_image")
show_with_matplotlib(lenna_bicubic_resize_image,"lenna_bicubic_resize_image")        

"""

############################################################################################## TEST ##############################################################################
