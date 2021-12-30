# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 21:19:23 2021

@author: 邵键帆
"""


"""
This program is aimed at contour detection
this program is based on paper: Topological Structural Analyssi of 
Digitized Binary Image by Border Following
the binary transformation we will apply otsu threshold transformation and adaptive gaussian threshold transformation
"""


#import relevant package
import numpy as np
import cv2 
from color_trans import bgr2gray,gray2bgr_show
from edge_detection import array_edge_detection
from convolution import show_with_matplotlib,convolution_trans,show_with_matplotlib_array,bgr_imread,image_merge
from threshold_segmentation import gray_scale_otsu_threshold


############################################################################################## Contour detection ##############################################################################


#define the function to capture the object contour
def image_contour_detection(image):
    gray_scale_array=bgr2gray(image)
    otsu_array=gray_scale_otsu_threshold(gray_scale_array)[:,:,0]
    binary_image=np.where(otsu_array==255,1,0)
    fake_kernel=np.ones((3,3),dtype="uint8")
    padding_binary_image=convolution_trans(binary_image,fake_kernel,1,1).pad_zero()
    image_row=padding_binary_image.shape[0]
    image_col=padding_binary_image.shape[1]
    NBD=1
    LNBD=1
    for i in range(image_row):
        LNBD=1
        for j in range(image_col):
            if padding_binary_image[i,j]==1 and padding_binary_image[i,j-1]==0:
                NBD += 1
                i2,j2=i,j-1
            elif padding_binary_image[i,j]>=1 and padding_binary_image[i,j+1]==0:
                NBD += 1
                i2,j2=i,j+1
                if padding_binary_image[i,j]>1:
                    LNBD=padding_binary_image[i,j]
            
            else:
                
                if padding_binary_image[i,j] !=1:
                    LNBD=abs(padding_binary_image[i,j])
                    
                continue
            
            
 
            
            if j2==j-1:
                clockwise_value_list=[padding_binary_image[i,j-1],padding_binary_image[i-1,j],padding_binary_image[i,j+1],padding_binary_image[i+1,j]]
                location_list=[[i,j-1],[i-1,j],[i,j+1],[i+1,j]]
            else:
                clockwise_value_list=[padding_binary_image[i,j+1],padding_binary_image[i+1,j],padding_binary_image[i,j-1],padding_binary_image[i-1,j]]
                location_list=[[i,j+1],[i+1,j],[i,j-1],[i-1,j]]
                
            
            
            if not any(clockwise_value_list):
                padding_binary_image[i,j]=-NBD
                if padding_binary_image[i,j] !=1:
                    LNBD=abs(padding_binary_image[i,j])
                    
                continue
            
            else:
                
                not_zero_location=clockwise_value_list.index(next(filter(lambda x: x!=0,clockwise_value_list)))
                
                location=location_list[not_zero_location]
                i1,j1=location[0],location[1]
                
                
            
            i2,j2=i1,j1
            i3,j3=i,j
            
            #clockwise_whole_value_list=[padding_binary_image[i,j+1],padding_binary_image[i+1,j],padding_binary_image[i,j-1],padding_binary_image[i-1,j]]
            #location_whole_list=[[i,j+1],[i+1,j],[i,j-1],[i-1,j]]
            
            i4=-3
            j4=-4
            kk=[]
            gg=[]
            while (i4!=i or j4!=j or i3!=i1 or j3!=j1):
                clockwise_whole_value_list=[padding_binary_image[i3,j3+1],padding_binary_image[i3+1,j3],padding_binary_image[i3,j3-1],padding_binary_image[i3-1,j3]]
                location_whole_list=[[i3,j3+1],[i3+1,j3],[i3,j3-1],[i3-1,j3]]
                kk.append(location_whole_list)
                #print(kk[0])
                clockwise_inv_whole_value_list=list(reversed(clockwise_whole_value_list))
                
                
                location_inv_whole_list=list(reversed(location_whole_list))
                index_i2_j2=location_inv_whole_list.index([i2,j2])
                
                if index_i2_j2==3:
                    index_i2_j2_add=0
                else:
                    index_i2_j2_add=index_i2_j2+1
                
                gg.append(location_inv_whole_list[index_i2_j2_add])
                #print(gg[0])
                clockwise_inv_whole_value_list_extend=clockwise_inv_whole_value_list.copy()
                clockwise_inv_whole_value_list_extend.extend(clockwise_inv_whole_value_list_extend[:index_i2_j2_add])
                
                location_inv_whole_list_extend=location_inv_whole_list.copy()
                location_inv_whole_list_extend.extend(location_inv_whole_list_extend[:index_i2_j2_add])
                
                clockwise_inv_whole_value_list_i4_j4=clockwise_inv_whole_value_list_extend[index_i2_j2_add:]
                
                location_inv_whole_list_i4_j4 =location_inv_whole_list_extend[index_i2_j2_add:]
                
                
                if any(clockwise_inv_whole_value_list_i4_j4):
                    
                    not_zero_i4_j4_location=clockwise_inv_whole_value_list_i4_j4.index(next(filter(lambda x: x!=0,clockwise_inv_whole_value_list_i4_j4)))
                    
                    i4,j4=location_inv_whole_list_i4_j4[not_zero_i4_j4_location][0],location_inv_whole_list_i4_j4[not_zero_i4_j4_location][1]
                    i3_j31_index=location_inv_whole_list_i4_j4.index([i3,j3+1])
                    
                    if i3_j31_index<not_zero_i4_j4_location and padding_binary_image[i3,j3+1]==0:
                        
                        padding_binary_image[i3,j3]=-NBD
                    elif i3_j31_index>not_zero_i4_j4_location and padding_binary_image[i3,j3]==1:
                       
                        padding_binary_image[i3,j3]=NBD
                    
                    if i4==i and j4==j and i3==i1 and j3==j1:
                        
                        if padding_binary_image[i,j] != 1:
                            LNBD=abs(padding_binary_image[i,j])
                    else:
                        
                        i2,j2=i3,j3
                        i3,j3=i4,j4
                        

    trans_padding_binary_image=np.where(padding_binary_image!=0,255,0)
    
    
    return trans_padding_binary_image[1:-1,1:-1]

############################################################################################## Contour detection ##############################################################################




############################################################################################## TEST ##############################################################################


lenna_image=cv2.imread("pics\\lenna.png")
lenna_contour_detection=image_contour_detection(lenna_image)
lenna_contour_detection_show=gray2bgr_show(lenna_contour_detection)
show_with_matplotlib(lenna_contour_detection_show,"lenna_contour_detection")
 

############################################################################################## TEST ##############################################################################
