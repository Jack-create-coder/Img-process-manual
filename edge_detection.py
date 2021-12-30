# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 16:45:12 2021

@author: 邵键帆
"""


"""
This program is aimed at edge detection
"""

#import the relevant package
from convolution import image_convolution_trans,show_with_matplotlib,convolution_trans,show_with_matplotlib_array,bgr_imread
from arithmetic_operation import saturation_array,saturation_image
import numpy as np
from matplotlib import pyplot as plt
from color_trans import bgr2gray,gray2bgr_show
import math
from arithmetic_operation import saturation_array
from typical_filter import gaussian_trans


############################################################################################## edge detection related filter ###########################################################################



#1.laplace filter

laplace_filter=np.array([[0,1,0],
                         [1,-4,1],
                         [0,1,0]])



#2.sobel sharpen filter

#(1) sobel gradient_x filter 

sobel_gx_filter=np.array([[-1,0,1],
                          [-2,0,2],
                          [-1,0,1]])

#(2) sobel gradient_y filter

sobel_gy_filter=np.array([[1,2,1],
                          [0,0,0],
                          [-1,-2,-1]])






############################################################################################## array edge detection ###########################################################################


#define a class including the relevant edge detection apporach against the target array
class array_edge_detection():
    
    #initialize the relevant parameters for the array edge detection
    def __init__(self,array,s,t):
        self.array=array
        self.s=s
        self.t=t


    #return a laplace filter tranforming array
    def laplace_trans(self):
        laplace_trans_class=convolution_trans(self.array,laplace_filter,self.s,self.t)
        laplace_trans_array=laplace_trans_class.convolution()
        laplace_trans_saturation_trans_array=saturation_array(laplace_trans_array)
        return laplace_trans_saturation_trans_array
     
    #return a sobel x-dimensional gradient transforming array
    def sobel_gx_trans(self):
        sobel_gx_trans_class=convolution_trans(self.array,sobel_gx_filter,self.s,self.t)
        sobel_gx_trans_array=sobel_gx_trans_class.convolution()
        sobel_gx_trans_saturation_trans_array=saturation_array(sobel_gx_trans_array)
        return sobel_gx_trans_saturation_trans_array
    
    #return a sobel y-dimensional gradient transforming array
    def sobel_gy_trans(self):
        sobel_gy_trans_class=convolution_trans(self.array,sobel_gy_filter,self.s,self.t)
        sobel_gy_trans_array=sobel_gy_trans_class.convolution()
        sobel_gy_trans_saturation_trans_array=saturation_array(sobel_gy_trans_array)
        return sobel_gy_trans_saturation_trans_array
    
    #return a soble filter transforming array
    def sobel_trans(self):
        sobel_gx_trans_class=convolution_trans(self.array,sobel_gx_filter,self.s,self.t)
        sobel_gx_trans_array=sobel_gx_trans_class.convolution()
        sobel_gy_trans_class=convolution_trans(self.array,sobel_gy_filter,self.s,self.t)
        sobel_gy_trans_array=sobel_gy_trans_class.convolution()
        sobel_trans_array=np.abs(sobel_gx_trans_array)+np.abs(sobel_gy_trans_array)
        sobel_trans_saturation_trans_array=saturation_array(sobel_trans_array)
        return sobel_trans_saturation_trans_array


############################################################################################## array edge detection ###########################################################################





############################################################################# canny edge detection ######################################################################################


"""

This section implement the canny edge detection for the gray-scale image

"""

#return the one-channel image=array (gray-scale image) sobel gradient and gradient degree
def sobel_intense_degree(array,s,t):

    sobel_gx_trans_class=convolution_trans(array,sobel_gx_filter,s,t)
    sobel_gx_trans_array=sobel_gx_trans_class.convolution()
    sobel_gy_trans_class=convolution_trans(array,sobel_gy_filter,s,t)
    sobel_gy_trans_array=sobel_gy_trans_class.convolution()
    sobel_intense_array=np.abs(sobel_gx_trans_array)+np.abs(sobel_gy_trans_array)
    sobel_degree_array=np.arctan2(sobel_gy_trans_array,sobel_gx_trans_array)*180/np.pi
    return sobel_intense_array,sobel_degree_array

#implement the non-maximum suppression(accurate degree-linear interploation) to the gary-scale image(one-channel image=array) transformed by the sobel filter(both-direction transformtion)
def non_maximum_suppression_accurate(array,s,t):
    sobel_intense_array,sobel_degree_array=sobel_intense_degree(array,s,t)
    fake_kernel=np.ones((3,3),dtype="uint8")
    
    padding_sobel_intense_array=convolution_trans(sobel_intense_array,fake_kernel,s,t).pad_zero()
    #print(padding_sobel_intense_array)
    for i in range(sobel_degree_array.shape[0]):
        for j in range(sobel_degree_array.shape[1]):
            sobel_intense=padding_sobel_intense_array[i+1,j+1]
            sobel_degree=sobel_degree_array[i,j]
            
            if (sobel_degree>=0 and sobel_degree <45) or (sobel_degree>=-180 and sobel_degree<-135):
                
                g_in_1=padding_sobel_intense_array[i+1,j+1+1]
                g_out_1=padding_sobel_intense_array[i+1-1,j+1+1]
                g_in_2=padding_sobel_intense_array[i+1,j+1-1]
                g_out_2=padding_sobel_intense_array[i+1+1,j+1-1]
                w=math.tan((sobel_degree/180)*math.pi)
                
                temp_1=w*g_in_1+(1-w)*g_out_1
                temp_2=w*g_in_2+(1-w)*g_out_2
                max_value=max([sobel_intense,temp_1,temp_2])
                
                if max_value != sobel_intense:
                    padding_sobel_intense_array[i+1,j+1]=0
                    #print(i,j)
                    #print(padding_sobel_intense_array[i+1,j+1])
            
            elif (sobel_degree>=45 and sobel_degree < 90) or (sobel_degree>=-135 and sobel_degree<-90):
                g_in_1=padding_sobel_intense_array[i+1-1,j+1]
                g_out_1=padding_sobel_intense_array[i+1-1,j+1+1]
                g_in_2=padding_sobel_intense_array[i+1+1,j+1]
                g_out_2=padding_sobel_intense_array[i+1+1,j+1-1]
                w=math.tan((sobel_degree/180)*math.pi)
                temp_1=w*g_in_1+(1-w)*g_out_1
                temp_2=w*g_in_2+(1-w)*g_out_2
                max_value=max([sobel_intense,temp_1,temp_2])
                if max_value != sobel_intense:
                    
                    padding_sobel_intense_array[i+1,j+1]=0               
            
            elif (sobel_degree>=90 and sobel_degree<145) or (sobel_degree >= -90 and sobel_degree<-45):
                g_in_1=padding_sobel_intense_array[i+1-1,j+1]
                g_out_1=padding_sobel_intense_array[i+1-1,j+1-1]
                g_in_2=padding_sobel_intense_array[i+1+1,j+1]
                g_out_2=padding_sobel_intense_array[i+1+1,j+1+1]
                w=math.tan(((sobel_degree-90)/180)*math.pi)
                temp_1=w*g_in_1+(1-w)*g_out_1
                temp_2=w*g_in_2+(1-w)*g_out_2
                max_value=max([sobel_intense,temp_1,temp_2])
                if max_value != sobel_intense:
                    padding_sobel_intense_array[i+1,j+1]=0
            
            else:
                g_in_1=padding_sobel_intense_array[i+1,j+1-1]
                g_out_1=padding_sobel_intense_array[i+1-1,j+1-1]
                g_in_2=padding_sobel_intense_array[i+1,j+1+1]
                g_out_2=padding_sobel_intense_array[i+1+1,j+1+1]
                w=math.tan(((180-sobel_degree)/180)*math.pi)
                temp_1=w*g_in_1+(1-w)*g_out_1
                temp_2=w*g_in_2+(1-w)*g_out_2
                max_value=max([sobel_intense,temp_1,temp_2])
                if max_value != sobel_intense:
                    padding_sobel_intense_array[i+1,j+1]=0
        
                    
        
    return padding_sobel_intense_array[1:-1,1:-1]

#implement the non-maximum suppression(not accurate using the pixed on the neighbor point) to the gary-scale image(one-channel image=array) transformed by the sobel filter(both-direction transformtion)
def non_maximum_suppression_rough(array,s,t):
    sobel_intense_array,sobel_degree_array=sobel_intense_degree(array,s,t)
    fake_kernel=np.ones((3,3),dtype="uint8")
    
    padding_sobel_intense_array=convolution_trans(sobel_intense_array,fake_kernel,s,t).pad_zero()
    #print(padding_sobel_intense_array)
    for i in range(sobel_degree_array.shape[0]):
        for j in range(sobel_degree_array.shape[1]):
            sobel_intense=padding_sobel_intense_array[i+1,j+1]
            sobel_degree=sobel_degree_array[i,j]

            if (sobel_degree>=0 and sobel_degree <45) or (sobel_degree>=-180 and sobel_degree<-135):
                
                if (sobel_degree>=0 and sobel_degree<22.5) or (sobel_degree>=-180 and sobel_degree<=-167.5):
                    temp_1=padding_sobel_intense_array[i+1,j+1+1]
                    temp_2=padding_sobel_intense_array[i+1,j+1-1]
                    max_value=max([sobel_intense,temp_1,temp_2])
                    if max_value != sobel_intense:
                        padding_sobel_intense_array[i+1,j+1]=0
                else:
                    temp_1=padding_sobel_intense_array[i+1-1,j+1+1]
                    temp_2=padding_sobel_intense_array[i+1+1,j+1-1]
                    max_value=max([sobel_intense,temp_1,temp_2])
                    if max_value != sobel_intense:
                        padding_sobel_intense_array[i+1,j+1]=0                    
                    
                    

            
            elif (sobel_degree>=45 and sobel_degree < 90) or (sobel_degree>=-135 and sobel_degree<-90):
                
                if (sobel_degree>=45 and sobel_degree<67.5) or (sobel_degree>=-135 and sobel_degree<(-135+22.5)):
                    temp_1=padding_sobel_intense_array[i+1-1,j+1+1]
                    temp_2=padding_sobel_intense_array[i+1+1,j+1-1]
                    max_value=max([sobel_intense,temp_1,temp_2])
                    if max_value != sobel_intense:
                        padding_sobel_intense_array[i+1,j+1]=0
                else:
                    temp_1=padding_sobel_intense_array[i+1-1,j+1]
                    temp_2=padding_sobel_intense_array[i+1+1,j+1]
                    max_value=max([sobel_intense,temp_1,temp_2])
                    if max_value != sobel_intense:
                        padding_sobel_intense_array[i+1,j+1]=0                    
                    
                    
                    
            
            elif (sobel_degree>=90 and sobel_degree<145) or (sobel_degree >= -90 and sobel_degree<-45):
                
                if (sobel_degree>=90 and sobel_degree<90+22.5) or (sobel_degree >= -90 and sobel_degree<(-90+22.5)):
                    temp_1=padding_sobel_intense_array[i+1-1,j+1]
                    temp_2=padding_sobel_intense_array[i+1+1,j+1]
                    max_value=max([sobel_intense,temp_1,temp_2])
                    if max_value != sobel_intense:
                        padding_sobel_intense_array[i+1,j+1]=0
                else:
                    temp_1=padding_sobel_intense_array[i+1-1,j+1-1]
                    temp_2=padding_sobel_intense_array[i+1+1,j+1+1]
                    max_value=max([sobel_intense,temp_1,temp_2])
                    if max_value != sobel_intense:
                        padding_sobel_intense_array[i+1,j+1]=0
                        
                    
            
            else:
                
                
                if(sobel_degree>=145 and sobel_degree<(145+22.5)) or (sobel_degree>=-45 and sobel_degree <(-45+22.5)):
                    temp_1=padding_sobel_intense_array[i+1-1,j+1-1]
                    temp_2=padding_sobel_intense_array[i+1+1,j+1+1]
                    max_value=max([sobel_intense,temp_1,temp_2])
                    if max_value != sobel_intense:
                        padding_sobel_intense_array[i+1,j+1]=0
                else:
                    temp_1=padding_sobel_intense_array[i+1,j+1-1]
                    temp_2=padding_sobel_intense_array[i+1,j+1+1]
                    max_value=max([sobel_intense,temp_1,temp_2])
                    if max_value != sobel_intense:
                        padding_sobel_intense_array[i+1,j+1]=0

    
    return padding_sobel_intense_array[1:-1,1:-1]



#return the pixel frequency of gray-scale image (one channel image)
def image_frequency(image):
    value_list=np.unique(image)
    
    freq_list=[]
    image_pixel_sorted=np.unique(image)
    
    for i in np.unique(image)[::-1]:
        
        freq_list.append(np.sum(image==i))

    return value_list.tolist()[::-1],freq_list


#return the high-low threshold according to the whole image pixel(30%: high threshold ; 45%: low threshol)
def high_low_threshold(array,s,t):
    sobel_intense_array,sobel_degree_array=sobel_intense_degree(array,s,t)
    sobel_intense_list,sobel_freq_list=image_frequency(sobel_intense_array)
    sobel_freq_culmutive=[sum(sobel_freq_list[:i])/sum(sobel_freq_list) for i in range(len(sobel_freq_list))]
    high_value_list=[]
    high_freq_list=[]
    low_value_list=[]
    low_freq_list=[]
    for i in range(len(sobel_intense_list)):
        if sobel_freq_culmutive[i]<=0.3 and sobel_freq_culmutive[i+1]>=0.3:
            high_threshold=sobel_intense_list[i+1]
        
        if sobel_freq_culmutive[i]<=0.45 and sobel_freq_culmutive[i+1]>=0.45:
            low_threshold=sobel_intense_list[i+1]
    
    return high_threshold,low_threshold





#implement the whole-process canny transformation to the gray-scale image(one-channel image-array)
def array_canny_trans(array,s,t,gaussian_sigma_space=0.5,row_num=3,col_num=3):
    gaussian_array=gaussian_trans(array,row_num,col_num,gaussian_sigma_space,s,t)
    non_maximum_tran_array=non_maximum_suppression_rough(gaussian_array,s,t)
    high_threshold,low_threshold=high_low_threshold(gaussian_array,s,t)
    fake_kernel=np.ones((3,3),dtype="uint8")
    non_maximum_tran_padding=convolution_trans(non_maximum_tran_array,fake_kernel,s,t).pad_zero()
    for i in range(1,non_maximum_tran_padding.shape[0]-1):
        for j in range(1,non_maximum_tran_padding.shape[1]-1):
            if non_maximum_tran_padding[i,j]<=low_threshold:
                non_maximum_tran_padding[i,j]=0
            elif non_maximum_tran_padding[i,j]>low_threshold and non_maximum_tran_padding[i,j]<=high_threshold:
                if non_maximum_tran_padding[i-1,j-1] < high_threshold and non_maximum_tran_padding[i-1,j] < high_threshold and non_maximum_tran_padding[i-1,j+1] < high_threshold and non_maximum_tran_padding[i,j-1] < high_threshold and non_maximum_tran_padding[i,j+1] < high_threshold and non_maximum_tran_padding[i+1,j-1] < high_threshold and non_maximum_tran_padding[i+1,j] < high_threshold and non_maximum_tran_padding[i+1,j+1] < high_threshold:
                    non_maximum_tran_padding[i,j]=0
    
    return saturation_array(non_maximum_tran_padding)


#implement the whole-process canny transformation to the image(three-channel image)
def image_canny_trans(image,s,t,gaussian_sigma_space=0.5,row_num=3,col_num=3):
    array=bgr2gray(image)
    gaussian_array=gaussian_trans(array,row_num,col_num,gaussian_sigma_space,s,t)
    non_maximum_tran_array=non_maximum_suppression_rough(gaussian_array,s,t)
    high_threshold,low_threshold=high_low_threshold(gaussian_array,s,t)
    fake_kernel=np.ones((3,3),dtype="uint8")
    non_maximum_tran_padding=convolution_trans(non_maximum_tran_array,fake_kernel,s,t).pad_zero()
    for i in range(1,non_maximum_tran_padding.shape[0]-1):
        for j in range(1,non_maximum_tran_padding.shape[1]-1):
            if non_maximum_tran_padding[i,j]<=low_threshold:
                non_maximum_tran_padding[i,j]=0
            elif non_maximum_tran_padding[i,j]>low_threshold and non_maximum_tran_padding[i,j]<=high_threshold:
                if non_maximum_tran_padding[i-1,j-1] < high_threshold and non_maximum_tran_padding[i-1,j] < high_threshold and non_maximum_tran_padding[i-1,j+1] < high_threshold and non_maximum_tran_padding[i,j-1] < high_threshold and non_maximum_tran_padding[i,j+1] < high_threshold and non_maximum_tran_padding[i+1,j-1] < high_threshold and non_maximum_tran_padding[i+1,j] < high_threshold and non_maximum_tran_padding[i+1,j+1] < high_threshold:
                    non_maximum_tran_padding[i,j]=0
    
    return saturation_array(non_maximum_tran_padding)

                    


###################################################################################### Canny edge detection  #####################################################################
  
    



############################################################################################## color edge detection ###########################################################################




#define a class including the relevant edge detection apporach against the target image
class image_edge_detection():
    
    #initialize the relevant parameters for the image edge detection
    def __init__(self,image,s,t):
        self.image=image
        self.s=s
        self.t=t
        
        
    #return a laplace filter tranforming image
    def image_laplace_trans(self):
        
        laplace_trans_image=image_convolution_trans(self.image,laplace_filter,self.s,self.t)
        laplace_trans_saturation_trans_image=saturation_image(laplace_trans_image)
        return laplace_trans_saturation_trans_image

    #return a sobel x-dimensional gradient transforming image
    def image_sobel_gx_trans(self):
        sobel_gx_trans_image=image_convolution_trans(self.image,sobel_gx_filter,self.s,self.t)
        sobel_gx_trans_saturation_trans_image=saturation_image(sobel_gx_trans_image)
    
        return sobel_gx_trans_saturation_trans_image

    #return a sobel y-dimensional gradient transforming image
    def image_sobel_gy_trans(self):
        sobel_gy_trans_image=image_convolution_trans(self.image,sobel_gy_filter,self.s,self.t)
        sobel_gy_trans_saturation_trans_image=saturation_image(sobel_gy_trans_image)
    
        return sobel_gy_trans_saturation_trans_image

    #return a soble filter transforming image
    def image_sobel_trans(self):
        sobel_gx_trans_image_in=image_convolution_trans(self.image,sobel_gx_filter,self.s,self.t)
        sobel_gy_trans_image_in=image_convolution_trans(self.image,sobel_gy_filter,self.s,self.t)
        sobel_trans_image=np.abs(sobel_gx_trans_image_in)+np.abs(sobel_gy_trans_image_in)
        sobel_trans_saturation_trans_image=saturation_image(sobel_trans_image)
        
        return sobel_trans_saturation_trans_image




############################################################################################## edge detection ###########################################################################




############################################################################################## TEST ##############################################################################


"""
lenna_array=bgr_imread("pics/lenna.png")[:,:,0]
lenna_array_edge_detection=array_edge_detection(lenna_array,1,1)
lenna_array_laplace_trans=lenna_array_edge_detection.laplace_trans()
lenna_array_sobel_gx_trans=lenna_array_edge_detection.sobel_gx_trans()
lenna_array_sobel_gy_trans=lenna_array_edge_detection.sobel_gy_trans()
lenna_array_sobel_trans=lenna_array_edge_detection.sobel_trans()

show_with_matplotlib_array(lenna_array_laplace_trans,"lenna_array_laplace_trans")
show_with_matplotlib_array(lenna_array_sobel_gx_trans,"lenna_array_sobel_gx_trans")
show_with_matplotlib_array(lenna_array_sobel_gy_trans,"lenna_array_sobel_gy_trans")
show_with_matplotlib_array(lenna_array_sobel_trans,"lenna_array_sobel_trans")


"""

"""    

lenna_image=bgr_imread("pics\\lenna.png")
lenna_gray_array=bgr2gray(lenna_image)

lenna_gray_canny_trans=gray2bgr_show(array_canny_trans(lenna_gray_array,1,1))
lenna_canny_trans=gray2bgr_show(image_canny_trans(lenna_image,1,1))
show_with_matplotlib(lenna_canny_trans,"lenna_canny_trans")
show_with_matplotlib(lenna_gray_canny_trans,"lenna_gray_canny_trans")

"""

"""

lenna_image=bgr_imread("pics\\lenna.png")
lenna_image_edge_detection_class=image_edge_detection(lenna_image,1,1)
lenna_image_laplace_trans=lenna_image_edge_detection_class.image_laplace_trans()
lenna_image_sobel_gx_trans=lenna_image_edge_detection_class.image_sobel_gx_trans()
lenna_image_sobel_gy_trans=lenna_image_edge_detection_class.image_sobel_gy_trans()
lenna_image_sobel_trans=lenna_image_edge_detection_class.image_sobel_trans()

show_with_matplotlib(lenna_image_laplace_trans,"lenna_image_laplace_trans")
show_with_matplotlib(lenna_image_sobel_gx_trans,"lenna_image_sobel_gx_trans")
show_with_matplotlib(lenna_image_sobel_gy_trans,"lenna_image_sobel_gy_trans")
show_with_matplotlib(lenna_image_sobel_trans,"lenna_image_sobel_trans")

"""

############################################################################################## TEST ##############################################################################
