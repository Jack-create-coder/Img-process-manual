# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 13:05:44 2021

@author: 邵键帆
"""


import numpy as np
import pandas as pd
from convolution import show_with_matplotlib,show_with_matplotlib_array,bgr_imread,image_merge
import math
from arithmetic_operation import image_split,image_merge
from image_save import image2excel




######################################################### color_space transformation ####################################################


#transform the bgr-image into gray-scale image
def bgr2gray(image):
    copy_image=image.copy()
       
    
    trans_image=copy_image[:,:,0]*0.114+copy_image[:,:,1]*0.5870+copy_image[:,:,2]*0.2989
        
    return trans_image.astype("uint8")

#display the gray-scale image in a bgr form
def gray2bgr_show(image):
    
    copy_image=image.copy()
    image_one=np.ones((image.shape[0],image.shape[1],3),"uint8")
    image_one[:,:,0],image_one[:,:,1],image_one[:,:,2]=copy_image,copy_image,copy_image
    trans_image=image_one.copy()
    return trans_image.astype("uint8")
    

#transform the bgr-image into hsv-form image
def bgr2hsv(image):
    max_image=image.max(axis=2)
    min_image=image.min(axis=2)
    image_copy=image/255.0
    b_image_copy,g_image_copy,r_image_copy=image_split(image_copy)
    max_image_copy=image_copy.max(axis=2)
    min_image_copy=image_copy.min(axis=2)
    hsv_v=max_image_copy
    hsv_s=np.divide((max_image-min_image).astype(float),max_image.astype(float),out=np.zeros_like(max_image).astype(float),where=max_image!=0)
    
    hsv_h=np.ones(hsv_v.shape)
    for i in range(hsv_v.shape[0]):
        for j in range(hsv_v.shape[1]):
            if max_image_copy[i,j]==min_image_copy[i,j]:
                hsv_h[i,j]=0
            elif min_image_copy[i,j]==b_image_copy[i,j]:
                hsv_h[i,j]=60+((60*(g_image_copy[i,j]-r_image_copy[i,j]))/(max_image_copy[i,j]-min_image_copy[i,j]))
            elif min_image_copy[i,j]==r_image_copy[i,j]:
                hsv_h[i,j]=180+((60*(b_image_copy[i,j]-g_image_copy[i,j]))/(max_image_copy[i,j]-min_image_copy[i,j]))
            else:
                hsv_h[i,j]=300+((60*(r_image_copy[i,j]-b_image_copy[i,j]))/(max_image_copy[i,j]-min_image_copy[i,j]))
    hsv_h_cv=hsv_h*(1/2)
    hsv_s_cv=hsv_s*255
    hsv_v_cv=hsv_v*255
    hsv_image_cv=image_merge([hsv_h_cv,hsv_s_cv,hsv_v_cv])
    
    return hsv_image_cv.astype("uint8")

#transform the hsv-form image into the bgr-form image 
def hsv2bgr(image):
    h,s,v=image_split(image)
    s=s/255
    v=v/255
    
    h_judge=np.mod((np.floor(h/60)),6)
    print(h_judge)
    f=(h/60)-h_judge
    p=np.multiply(v,1-s)
    q=np.multiply(v,1-np.multiply(f,s))
    t=np.multiply(v,(1-np.multiply(1-f,s)))
    trans_image=np.ones(image.shape)
    for i in range(trans_image.shape[0]):
        for j in range(trans_image.shape[1]):
            if h_judge[i,j]==0:
                trans_image[i,j,:]=np.array([v[i,j],t[i,j],p[i,j]])
            elif h_judge[i,j]==1:
                trans_image[i,j,:]=np.array([q[i,j],v[i,j],p[i,j]])
            elif h_judge[i,j]==2:
                trans_image[i,j,:]=np.array([p[i,j],v[i,j],t[i,j]])
            elif h_judge[i,j]==3:
                trans_image[i,j,:]=np.array([p[i,j],q[i,j],v[i,j]])
            elif h_judge[i,j]==4:
                trans_image[i,j,:]=np.array([t[i,j],p[i,j],v[i,j]])
            else:
                trans_image[i,j,:]=np.array([v[i,j],p[i,j],q[i,j]])
    trans_image_scale=(trans_image*255)[:,:,::-1]
    
    return trans_image_scale.astype("uint8")









######################################################### color_space transformation ####################################################


#########################################################  gray-scale transformation  ####################################################


class gray_image_trans():
    
    def __init__(self,gray_image):
        self.gray_image=gray_image
        
        
    def complementary_gray(self):
        trans_image=255-self.gray_image
        return trans_image


    def log_gray(self):
        trans_image=np.log(1+self.gray_image)*(255/math.log(1+255))
        return trans_image.astype("uint8")

    def gamma_gray(self,gamma=1.7):
        trans_image=np.power(self.gray_image,gamma)*(255/(255**gamma))
        return trans_image.astype("uint8")





######################################################### gray-scale transformation  ####################################################




######################################################### color_range transform  ####################################################


"""
bgr range:[0,255]
hsv range:h-[0,180],s-[0,255],v-[0,255]

"""

class image_range_capture():
    
    def __init__(self,lower_array,upper_array,image):
        self.lower_array=lower_array
        self.upper_array=upper_array
        self.image=image

    def image_range_color(self):
        capture_image=np.ones(self.image.shape,dtype="uint8")*255
        lower_0,lower_1,lower_2=self.lower_array
        upper_0,upper_1,upper_2=self.upper_array
        image_ndim=self.image.ndim
        one_channel=self.image[:,:,0]
    
        for i in range(one_channel.shape[0]):
            for j in range(one_channel.shape[1]):
                if (lower_0<self.image[i,j,0]<upper_0) and (lower_1<self.image[i,j,1]<upper_1) and (lower_2<self.image[i,j,2]<upper_2):
                    capture_image[i,j,:]=self.image[i,j,:]
        return capture_image

    def image_range_black(self):
        capture_image=np.ones(self.image.shape,dtype="uint8")*0
        lower_0,lower_1,lower_2=self.lower_array
        upper_0,upper_1,upper_2=self.upper_array
        image_ndim=self.image.ndim
        one_channel=self.image[:,:,0]
    
        for i in range(one_channel.shape[0]):
            for j in range(one_channel.shape[1]):
                if (lower_0<self.image[i,j,0]<upper_0) and (lower_1<self.image[i,j,1]<upper_1) and (lower_2<self.image[i,j,2]<upper_2):
                    capture_image[i,j,:]=np.array([255,255,255])
        return capture_image




######################################################### color_range transform  ####################################################
    


######################################################### color maps transform  ####################################################

"""
create a 3d look-up-table(LUT) for color maps transformation
"""

#return a (255,255,255) look-up-table
def complete_lut():
    complete_lut=np.ones((int(256*math.sqrt(256)),int(256*math.sqrt(256)),3),dtype="uint8")
    for i in range(complete_lut.shape[0]):
        for j in range(complete_lut.shape[1]):
            
            b_i=i//256
            
            b_j=j//256
            
            b=int(b_i*int(math.sqrt(256))+b_j)
            
            g=int(i % 256)
        
            r=int(j % 256)
            
            complete_lut[i,j,:]=np.array([b,g,r])
    return complete_lut


#return a reshaped look-up-table for color mpas transformation(the pixel will be compressed)
def complete_lut_trans(b_reshape,g_reshape,r_reshape):
    
    complete_lut=np.ones((int(256*math.sqrt(256)),int(256*math.sqrt(256)),3),dtype="uint8")
    b_scale=math.sqrt(256/b_reshape)
    g_scale=int(256/g_reshape)
    r_scale=int(256/r_reshape)
    for i in range(complete_lut.shape[0]):
        for j in range(complete_lut.shape[1]):
            b_i=int(((i//256)//b_scale)*b_scale)
            b_j=int(((j//256)//b_scale)*b_scale)
            b=int(b_i*16+b_j)
            
            g=int(((i%256)//g_scale)*g_scale)
            r=int(((j%256)//r_scale)*r_scale)
            
            complete_lut[i,j,:]=np.array([b,g,r])
            
    return complete_lut



#return a image transformed by the lut_template(the template needed to be created)
def image_lut_trans(image,lut_size):
    image_trans=image.copy()
    b,g,r=image_split(image)
    complete_lut_trans_image=complete_lut_trans(reshape_resize)
    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            b_i=b[i,j]//16
            b_j=b[i,j] % 16
            
            g_trans=b_i*256+g[i,j]
            r_trans=b_j*256+r[i,j]
            image_trans[i,j,:]=complete_lut_trans_image[g_trans,r_trans,:]
    return image_trans




#return a image transformed by the lut_template(the template is existing)
def image_lut_trans_with_template(image,template):
    image_trans=image.copy()
    b,g,r=image_split(image)
    complete_lut_trans_image=template
    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            b_i=b[i,j]//16
            b_j=b[i,j] % 16
            
            g_trans=b_i*256+g[i,j]
            r_trans=b_j*256+r[i,j]
            image_trans[i,j,:]=complete_lut_trans_image[g_trans,r_trans,:]
    return image_trans



######################################################### color maps transform  ####################################################





############################################################################################## TEST ##############################################################################



"""

lenna_image=bgr_imread("pics\\lenna.png")
lenna_gray_image=bgr2gray(lenna_image)
lenna_gray_image_trans=gray_image_trans(lenna_gray_image)
lenna_complementary_gray=gray2bgr_show(lenna_gray_image_trans.complementary_gray())
lenna_log_gray=gray2bgr_show(lenna_gray_image_trans.log_gray())
lenna_gamma_gray=gray2bgr_show(lenna_gray_image_trans.gamma_gray())

show_with_matplotlib(lenna_complementary_gray,"lenna_complementary_gray")
show_with_matplotlib(lenna_log_gray,"lenna_log_gray")
show_with_matplotlib(lenna_gamma_gray,"lenna_gamma_gray")

"""

"""

lenna_image=bgr_imread("pics\\lenna.png")
skin_1_image=bgr_imread("pics\\kids.png")
skin_1_hsv_image=bgr2hsv(skin_1_image)

lower_array_1=np.array([0,48,80])
upper_array_1=np.array([20,255,255])
kids_capture_image_class=image_range_capture(lower_array_1,upper_array_1,skin_1_hsv_image)
kids_capture_color=kids_capture_image_class.image_range_color()
kids_capture_black=kids_capture_image_class.image_range_black()
show_with_matplotlib(skin_1_image,"kids_original_skin_detect")
show_with_matplotlib(skin_1_hsv_image,"kids_hsv_image_skin_detect")
show_with_matplotlib(kids_capture_color,"kids_capture_image_color_skin_detect")
show_with_matplotlib(kids_capture_black,"kids_capture_image_black_skin_detect")

"""


"""
complete_lut_trans_4_4_4=complete_lut_trans(4,4,4)
complete_lut_trans_16_16_16=complete_lut_trans(16,16,16)
complete_lut_trans_16_64_256=complete_lut_trans(16,64,256)
np.save("lut_template\\complete_lut_trans_4_4_4.npy",complete_lut_trans_4_4_4)
np.save("lut_template\\complete_lut_trans_16_16_16",complete_lut_trans_16_16_16)
np.save("lut_template\\complete_lut_trans_16_16_16",complete_lut_trans_16_64_256)

"""



"""

skin_1_image=bgr_imread("pics\\kids.png")

complete_lut_trans_4_4_4=np.load("lut_template\\complete_lut_trans_4_4_4.npy")
complete_lut_trans_16_16_16=np.load("lut_template\\complete_lut_trans_16_16_16.npy")


skin_1_lut_trans_with_template_4_4_4=image_lut_trans_with_template(skin_1_image,complete_lut_trans_4_4_4)
skin_1_lut_trans_with_template_16_16_16=image_lut_trans_with_template(skin_1_image,complete_lut_trans_16_16_16)

show_with_matplotlib(complete_lut_trans_4_4_4,"complete_lut_trans_4_4_4")
show_with_matplotlib(complete_lut_trans_16_16_16,"complete_lut_trans_16_16_16")

show_with_matplotlib(skin_1_lut_trans_with_template_4_4_4,"skin_1_lut_trans_with_template_4_4_4")
show_with_matplotlib(skin_1_lut_trans_with_template_16_16_16,"skin_1_lut_trans_with_template_16_16_16")

"""

############################################################################################## TEST ##############################################################################
