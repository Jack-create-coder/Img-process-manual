# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 22:59:09 2021

@author: 邵键帆
"""



"""
This program is aimed at thresholding techniques for
image segmentation

Note: gaussian filter: sigma_space - [0,1]   gaussian adaptive threshold: sigma_space - more than 1

But if you want to implement gaussian filter to the threshold, the filter sigma_space will be more than 1 
"""

#import the relevant package
import numpy as np

from convolution import show_with_matplotlib_array,show_with_matplotlib
from color_trans import bgr2gray,gray2bgr_show
from arithmetic_operation import image_split,image_merge
from convolution import convolution_trans,bgr_imread
from typical_filter import normal_trans,gaussian_kernel,bilateral_trans,gaussian_trans
#################################################################### gray-scale threshold ########################################


#return the otsu optimal threshold value using the Maximum class variance method
def otsu_optimal_threshold(one_channel_image):
    threshold_value=0
    n=one_channel_image.shape[0]*one_channel_image.shape[1]
    threshold_value_list=[]
    variation_value_list=[]
    for threshold_value in range(256):
        font_n0=np.count_nonzero(one_channel_image>threshold_value)
        font_n1=np.count_nonzero(one_channel_image<=threshold_value)
        if font_n0==0:
            font_u0=0
        else:
            font_u0=(np.sum(np.where(one_channel_image>threshold_value,one_channel_image,0)))/font_n0
        if font_n1==0:
            font_u1=0
        else:
            font_u1=(np.sum(np.where(one_channel_image<=threshold_value,one_channel_image,0)))/font_n1
        partion_w0=font_n0/n
        partion_w1=font_n1/n
        variation_value=partion_w0*partion_w1*((font_u0-font_u1)**2)
        threshold_value_list.append(threshold_value)
        variation_value_list.append(variation_value)
    variation_value_max_index=variation_value_list.index(max(variation_value_list))
    return threshold_value_list[variation_value_max_index]



"""
return triangle threshold: error remained
"""
#return the point to the line which is determined by two points
def get_distance_from_point_to_line(point, line_point1, line_point2):
    
    if line_point1 == line_point2:
        point_array = np.array(point)
        point1_array = np.array(line_point1)
        return np.linalg.norm(point_array -point1_array )

    A = line_point2[1] - line_point1[1]
    B = line_point1[0] - line_point2[0]
    C = (line_point1[1] - line_point2[1]) * line_point1[0] + \
        (line_point2[0] - line_point1[0]) * line_point1[1]
    
    distance = np.abs(A * point[0] + B * point[1] + C) / (np.sqrt(A**2 + B**2))
    return distance

#return the pixel frequency of image
def image_frequency(image):
    value_list=np.unique(image)
    
    freq_list=[]
    
    for i in np.unique(image):
        freq_list.append(np.sum(image==i))

    return value_list.tolist(),freq_list

#return the optimal threshold value of triangle method
def triangle_optimal_threshold(image):
    value_list,freq_list=image_frequency(image)
    freq_max_index,freq_min_index=freq_list.index(max(freq_list)),freq_list.index(min(freq_list))
    freq_max_value,freq_min_value=value_list[freq_max_index],value_list[freq_min_index]
    freq_max,freq_min=max(freq_list),min(freq_list)

    freq_value_region_list=[]
    distance_list=[]
    if freq_max_value>int(255/2):
        for i in range(freq_min_value+1,freq_max_value):
            if i in value_list:
                freq=freq_list[value_list.index(i)]
                distance=get_distance_from_point_to_line([i,freq],[freq_min_value,freq_min],[freq_max_value,freq_max])
                freq_value_region_list.append(i)
                distance_list.append(distance)
    else:
        for i in range(freq_max_value+1,freq_min_value):
            if i in value_list:
                freq=freq_list[value_list.index(i)]
                distance=get_distance_from_point_to_line([i,freq],[freq_min_value,freq_min],[freq_max_value,freq_max])
                freq_value_region_list.append(i)
                distance_list.append(distance)
    
    return freq_value_region_list[distance_list.index(max(distance_list))]


  
"""
return triangle threshold: error remained
"""          
    

#return the normal threshold transformation of the gray-scale image(incuding binary\binary inverse\truncation\zero\to_zero) 
#return gray-bgr_show format image(three channels image)
class gray_scale_threshold_trans():
    
    def __init__(self,image,threshold_value):
        #image: gray-scale image(one-channel image)
        #threshold_value: 
        
        self.image=image
        self.threshold_value=threshold_value
    
    #binary threshold methodology (less than threshold:0  more than threshold:255)
    def gray_scale_binary_threshold(self):
        binary_array=np.where(self.image>self.threshold_value,255,0)
        binary_image=gray2bgr_show(binary_array)
        return binary_image
    #inverse binary threshold methodlogy (less than threshold: 255; more than threshold: 0)
    def gray_scale_binary_inv_threshold(self):
        binary_array=np.where(self.image>self.threshold_value,0,255)
        binary_inv_image=gray2bgr_show(binary_array)
        return binary_inv_image
    
    #truncation threshold methodology (more than threshold value: threshold value, less than threshold value: no change)
    def gray_scale_truncation_threshold(self):
        binary_array=np.where(self.image>self.threshold_value,self.threshold_value,self.image)
        binary_image=gray2bgr_show(binary_array)
        return binary_image

    #to zero inverse methodology (more than threshold value: 0; less than threshold value: no change)
    def gray_scale_tozero_inv_threshold(self):
        binary_array=np.where(self.image>self.threshold_value,0,self.image)
        binary_image=gray2bgr_show(binary_array)
        return binary_image

    #to zero methodology (less than threshold value: 0; more than threshold value: no change)
    def gray_scale_tozero_threshold(self):
        binary_array=np.where(self.image<self.threshold_value,0,self.image)
        binary_image=gray2bgr_show(binary_array)
        return binary_image


#implement the otsu optimal threshold value to the binary threshold methodology
def gray_scale_otsu_threshold(image):
    otus_threshold_value=otsu_optimal_threshold(image)
    otsu_array=np.where(image>otus_threshold_value,255,0)
    otsu_image=gray2bgr_show(otsu_array)
    return otsu_image






#remain error
class gray_scale_triangle_threshold_trans():
    
    def __init__(self,image):
        self.image=image
        
    
    def triangle_threshold(self):
        return triangle_optimal_threshold(self.image)
    
    
    def gray_scale_binary_triangle_threshold(self):
        triangle_threshold_class=gray_scale_threshold_trans(self.image,self.triangle_threshold())
        return triangle_threshold_class.gray_scale_binary_threshold()
    
    def gray_scale_binary_inv_triangle_threshold(self):
        triangle_threshold_class=gray_scale_threshold_trans(self.image,self.triangle_threshold())
        return triangle_threshold_class.gray_scale_binary_inv_threshold()    
    
    def gray_scale_truncation_triangle_threshold(self):
        triangle_threshold_class=gray_scale_threshold_trans(self.image,self.triangle_threshold())
        return triangle_threshold_class.gray_scale_truncation_threshold()
    
    def gray_scale_tozero_inv_triangle_threshold(self):
        triangle_threshold_class=gray_scale_threshold_trans(self.image,self.triangle_threshold())
        return triangle_threshold_class.gray_scale_tozero_inv_threshold()    
    
    def gray_scale_tozero_triangle_threshold(self):
        triangle_threshold_class=gray_scale_threshold_trans(self.image,self.triangle_threshold())
        return triangle_threshold_class.gray_scale_tozero_threshold()    
    



# implement the adaptive methodology into the threshold transformation(incuding: mean\median\gaussian threshold) and return gray-bgr_display image(three channels image)
class gray_scale_adaptive_trans():
    
    def __init__(self,image,kernel_size,delta=30,s=1,t=1):
        #image: gray_scale image (one-channel image)
        self.image=image
        self.kernel_size=kernel_size
        self.delta=delta
        self.s=s
        self.t=t

    def gray_scale_adaptive_mean_threshold(self):
        gray_scale_normal_trans=normal_trans(self.image,self.kernel_size,self.kernel_size,self.s,self.t)
        mean_threshold_array=gray_scale_normal_trans.average_trans()   
        gap_array=self.image-self.delta-mean_threshold_array
        binary_array=np.where(gap_array>0,255,0)
        binary_inv_array=np.where(gap_array>0,0,255)
        binary_image=gray2bgr_show(binary_array)
        binary_inv_image=gray2bgr_show(binary_inv_array)
        return binary_image,binary_inv_image

    def gray_scale_adaptive_median_threshold(self):
        gray_scale_normal_trans=normal_trans(self.image,self.kernel_size,self.kernel_size,self.s,self.t)
        median_threshold_array=gray_scale_normal_trans.median_trans()   
        gap_array=self.image-self.delta-median_threshold_array
        binary_array=np.where(gap_array>0,255,0)
        binary_inv_array=np.where(gap_array>0,0,255)
        binary_image=gray2bgr_show(binary_array)
        binary_inv_image=gray2bgr_show(binary_inv_array)
        return binary_image,binary_inv_image

    def gray_scale_adaptive_gaussian_threshold(self,sigma_space=5):
        gaussian_kernel_1=gaussian_kernel(self.kernel_size,self.kernel_size,sigma_space).gaussian_kernel_float()
        gaussian_trans_array=convolution_trans(self.image,gaussian_kernel_1,self.s,self.t).convolution()
        
    
        gap_array=self.image-self.delta-gaussian_trans_array
        binary_array=np.where(gap_array>0,255,0)
        binary_inv_array=np.where(gap_array>0,0,255)
        binary_image=gray2bgr_show(binary_array)
        binary_inv_image=gray2bgr_show(binary_inv_array)
        return binary_image,binary_inv_image






#capture the regional image based on the otsu threshold segmentation
def color_capture_gray_scale_otsu_threshold(image):
    gray_image=bgr2gray(image)
    otsu_capture_image=gray_scale_otsu_threshold(gray_image)
    judge_capture_image=np.where(otsu_capture_image[:,:,0]==255,1,0)
    
    b_capture=np.multiply(image[:,:,0],judge_capture_image)
    g_capture=np.multiply(image[:,:,1],judge_capture_image)
    r_capture=np.multiply(image[:,:,2],judge_capture_image)
    capture_image=image_merge([b_capture,g_capture,r_capture])
    return capture_image

#capture the regional image based on the adaptive gaussian threshold segmentation
def color_capture_gray_scale_adaptive_gaussian_threshold(image,kernel_size):
    gray_image=bgr2gray(image)
    adaptive_gaussian_capture_image,_=gray_scale_adaptive_trans(gray_image,kernel_size).gray_scale_adaptive_gaussian_threshold()
    
    judge_capture_image=np.where(adaptive_gaussian_capture_image[:,:,0]==255,1,0)
    b_capture=np.multiply(image[:,:,0],judge_capture_image)
    g_capture=np.multiply(image[:,:,1],judge_capture_image)
    r_capture=np.multiply(image[:,:,2],judge_capture_image)
    capture_image=image_merge([b_capture,g_capture,r_capture])
    return capture_image



#this section(small invoation) is aimed at integrating the gaussain and bilateral filters and threshold technique
#for the color image with noise, the filtering technique for wipe off the noise.


#noise filter with the adaptive gaussian demonstrated the bad effect of removing noise and threshold segmentation
def color_image_adaptive_gaussian_threshold_gaussian_filter(image,filter_size,filter_sigma_space,additive_kernel_size,additive_delta,additive_sigma):
    gaussian_filter=gaussian_kernel(filter_size,filter_size,filter_sigma_space).gaussian_kernel_float()
    gray_image=bgr2gray(image)
    gray_image_gaussian_filter_trans=convolution_trans(gray_image,gaussian_filter,1,1).convolution()
    gray_image_adaptive_trans_class=gray_scale_adaptive_trans(gray_image,additive_kernel_size,additive_delta)
    gray_image_adaptive_gaussian_threshold,gray_image_adaptive_gaussian_threshold_inv=gray_image_adaptive_trans_class.gray_scale_adaptive_gaussian_threshold(additive_sigma)
    return gray_image_adaptive_gaussian_threshold,gray_image_adaptive_gaussian_threshold_inv


#the best effect of removing the noise:integrating the gaussian filter (first step) and otsu threshold methodology to implement the threshold operation
def color_image_otsu_threshold_gaussian_filter(image,filter_row,filter_col,filter_sigma_space,s=1,t=1):
    gray_image=bgr2gray(image)
    gray_image_gaussian_trans=gaussian_trans(gray_image,filter_row,filter_col,filter_sigma_space,s,t)
    gray_image_gaussian_filter_otsu_threshold=gray_scale_otsu_threshold(gray_image_gaussian_trans)
    return gray_image_gaussian_filter_otsu_threshold


#bad effect of removing the noise: integrating the bilateral filter (first step) and otsu threshold methodology to implement the threshold operation
def color_image_otsu_threshold_bilateral_filter(image,filter_row,filter_col,sigma_space,sigma_color,s=1,t=1):
    gray_image=bgr2gray(image)
    gray_image_bilateral_trans=bilateral_trans(gray_image,filter_row,filter_col,sigma_space,s,t).bilateral_convolution(sigma_color)
    gray_image_bilateral_filter_otsu_threshold=gray_scale_otsu_threshold(gray_image_bilateral_trans)
    return gray_image_bilateral_filter_otsu_threshold


    
"""
This can remaind as a question to find the appropriate parameters for bilateral threshodl

It's difficult to find the appropriate parameters to implement the bilateral threshold

def gray_scale_adaptive_bilateral_threshold(image,kernel_size,sigma_space,sigma_color,delta=30,s=1,t=1):
    gray_scale_bilateral_trans=bilateral_trans(image,kernel_size,kernel_size,sigma_space,s,t)
    bilateral_threshold_array=gray_scale_bilateral_trans.bilateral_convolution(sigma_color)  
    gap_array=image-30-bilateral_threshold_array
    binary_array=np.where(gap_array>0,255,0)
    binary_inv_array=np.where(gap_array>0,0,255)
    binary_image=gray2bgr_show(binary_array)
    binary_inv_image=gray2bgr_show(binary_inv_array)
    return binary_image,binary_inv_image

"""


#################################################################### gray-scale threshold ########################################




#################################################################### color-scale threshold ########################################


## ! This section will act as an experimental program, the threshold transformation is normally applied in binary image([0,255])
## And at the same time implement the same solution in the color image is the unmature solution
#return the image transformed by the threshold methodology (input shape:3 channel image / output shape: 3 channel image)
class color_threshold_trans():
    
    #initialize the relevant parameters for the color_threshold_trans class
    def __init__(self,image,thre_b,thre_g,thre_r):
        #image: input image (three channels image)
        #thre_b,thre_g,thre_r: the individual threshold value for different channel image
        self.image=image
        self.thre_b=thre_b
        self.thre_g=thre_g
        self.thre_r=thre_r
    
    
    #return the image transformed by the binary methodology(refer to the aforementioned in gray-scale transformation )
    def color_binary_threshold(self):
        b,g,r=image_split(self.image)
        b_thre=np.where(b>self.thre_b,255,0)
        g_thre=np.where(g>self.thre_g,255,0)
        r_thre=np.where(r>self.thre_r,255,0)
        binary_image=image_merge([b_thre,g_thre,r_thre])
        return binary_image

    #refer to the aforementioned in gray-scale transformation
    def color_binary_inv_threshold(self):
        b,g,r=image_split(self.image)
        b_thre=np.where(b>self.thre_b,0,255)
        g_thre=np.where(g>self.thre_g,0,255)
        r_thre=np.where(r>self.thre_r,0,255)
        binary_inv_image=image_merge([b_thre,g_thre,r_thre])
        return binary_inv_image
    #refer to the aforementioned in gray-scale transformation
    def color_truncation_threshold(self):
        b,g,r=image_split(self.image)
        b_thre=np.where(b>self.thre_b,self.thre_b,b)
        g_thre=np.where(g>self.thre_g,self.thre_g,g)
        r_thre=np.where(r>self.thre_r,self.thre_r,r)
        truncation_image=image_merge([b_thre,g_thre,r_thre])
        return truncation_image
    #refer to the aforementioned in gray-scale transformation
    def color_tozero_inv_threshold(self):
        b,g,r=image_split(self.image)
        b_thre=np.where(b>self.thre_b,0,b)
        g_thre=np.where(g>self.thre_g,0,g)
        r_thre=np.where(r>self.thre_r,0,r)
        tozero_inv_image=image_merge([b_thre,g_thre,r_thre])
        return tozero_inv_image
    #refer to the aforementioned in gray-scale transformation
    def color_tozero_threshold(self):
        b,g,r=image_split(self.image)
        b_thre=np.where(b<self.thre_b,0,b)
        g_thre=np.where(g<self.thre_g,0,g)
        r_thre=np.where(r<self.thre_r,0,r)
        tozero_inv_image=image_merge([b_thre,g_thre,r_thre])
        return tozero_inv_image


#use the otsu optimal threshold values to implement the basic threshold transformation
class color_otsu_threshold_trans():
    
    def __init__(self,image):
        self.image=image
    
    def otsu_threshold(self):
        return otsu_optimal_threshold(self.image[:,:,0]),otsu_optimal_threshold(self.image[:,:,1]),otsu_optimal_threshold(self.image[:,:,2])
    
    def color_binary_otsu_threshold(self):
        b_otsu,g_otsu,r_otsu=self.otsu_threshold()
        otsu_threshold_class=color_threshold_trans(self.image,b_otsu,g_otsu,r_otsu)
        return otsu_threshold_class.color_binary_threshold()

    def color_binary_inv_otsu_threshold(self):
        b_otsu,g_otsu,r_otsu=self.otsu_threshold()
        otsu_threshold_class=color_threshold_trans(self.image,b_otsu,g_otsu,r_otsu)
        return otsu_threshold_class.color_binary_inv_threshold()

    def color_truncation_otsu_threshold(self):
        b_otsu,g_otsu,r_otsu=self.otsu_threshold()
        otsu_threshold_class=color_threshold_trans(self.image,b_otsu,g_otsu,r_otsu)
        return otsu_threshold_class.color_truncation_threshold()


    def color_tozero_inv_otsu_threshold(self):
        b_otsu,g_otsu,r_otsu=self.otsu_threshold()
        otsu_threshold_class=color_threshold_trans(self.image,b_otsu,g_otsu,r_otsu)
        return otsu_threshold_class.color_tozero_inv_threshold()

    
    def color_tozero_otsu_threshold(self):
        b_otsu,g_otsu,r_otsu=self.otsu_threshold()
        otsu_threshold_class=color_threshold_trans(self.image,b_otsu,g_otsu,r_otsu)
        return otsu_threshold_class.color_tozero_threshold()

#extend the gray_scale image(one-channel image) adaptive methodology to the color image(3 channels image)
class color_adaptive_trans():
    
    def __init__(self,image,kernel_size,delta=30,s=1,t=1):
        self.image=image
        self.kernel_size=kernel_size
        self.delta=delta
        self.s=s
        self.t=t
        
    
    def color_adaptive_mean_threshold(self):
        b,g,r=image_split(self.image)
        b_adaptive_trans=gray_scale_adaptive_trans(b,self.kernel_size,self.delta,self.s,self.t)
        g_adaptive_trans=gray_scale_adaptive_trans(g,self.kernel_size,self.delta,self.s,self.t)
        r_adaptive_trans=gray_scale_adaptive_trans(r,self.kernel_size,self.delta,self.s,self.t)
        b_mean_1,b_mean_2=b_adaptive_trans.gray_scale_adaptive_mean_threshold()
        g_mean_1,g_mean_2=g_adaptive_trans.gray_scale_adaptive_mean_threshold()
        r_mean_1,r_mean_2=r_adaptive_trans.gray_scale_adaptive_mean_threshold()
        
        return image_merge([b_mean_1[:,:,0],g_mean_1[:,:,0],r_mean_1[:,:,0]]),image_merge([b_mean_2[:,:,0],g_mean_2[:,:,0],r_mean_2[:,:,0]])

    def color_adaptive_median_threshold(self):
        b,g,r=image_split(self.image)
        b_adaptive_trans=gray_scale_adaptive_trans(b,self.kernel_size,self.delta,self.s,self.t)
        g_adaptive_trans=gray_scale_adaptive_trans(g,self.kernel_size,self.delta,self.s,self.t)
        r_adaptive_trans=gray_scale_adaptive_trans(r,self.kernel_size,self.delta,self.s,self.t)
        b_median_1,b_median_2=b_adaptive_trans.gray_scale_adaptive_median_threshold()
        g_median_1,g_median_2=g_adaptive_trans.gray_scale_adaptive_median_threshold()
        r_median_1,r_median_2=r_adaptive_trans.gray_scale_adaptive_median_threshold()
        
        return image_merge([b_median_1[:,:,0],g_median_1[:,:,0],r_median_1[:,:,0]]),image_merge([b_median_2[:,:,0],g_median_2[:,:,0],r_median_2[:,:,0]])

    def color_adaptive_gaussian_threshold(self,sigma_space=5):
        b,g,r=image_split(self.image)
        b_adaptive_trans=gray_scale_adaptive_trans(b,self.kernel_size,self.delta,self.s,self.t)
        g_adaptive_trans=gray_scale_adaptive_trans(g,self.kernel_size,self.delta,self.s,self.t)
        r_adaptive_trans=gray_scale_adaptive_trans(r,self.kernel_size,self.delta,self.s,self.t)
        b_gaussian_1,b_gaussian_2=b_adaptive_trans.gray_scale_adaptive_gaussian_threshold(sigma_space)
        g_gaussian_1,g_gaussian_2=g_adaptive_trans.gray_scale_adaptive_gaussian_threshold(sigma_space)
        r_gaussian_1,r_gaussian_2=r_adaptive_trans.gray_scale_adaptive_gaussian_threshold(sigma_space)
        
        return image_merge([b_gaussian_1[:,:,0],g_gaussian_1[:,:,0],r_gaussian_1[:,:,0]]),image_merge([b_gaussian_2[:,:,0],g_gaussian_2[:,:,0],r_gaussian_2[:,:,0]])





#################################################################### color-scale threshold ########################################



############################################################################################## TEST ##############################################################################


"""

leaf_noise_image = bgr_imread('pics\\leaf_noise.png')

leaf_otsu_threshold_gaussian_filter=color_image_otsu_threshold_gaussian_filter(leaf_noise_image,25,25,5)
show_with_matplotlib(leaf_otsu_threshold_gaussian_filter,"noise_leaf_otsu_threshold_gaussian_filter")

#bilateral will cost lots of time

#leaf_otsu_threshold_bilateral_filter=color_image_otsu_threshold_bilateral_filter(leaf_noise_image,25,25,5,0.5)
#show_with_matplotlib(leaf_noise_image,"noise_leaf_original_image")
#show_with_matplotlib(leaf_otsu_threshold_bilateral_filter,"noise_leaf_otsu_threshold_bilateral_filter")

"""


"""

lenna_image=bgr_imread("pics\\lenna.png")
lenna_image_color_capture_otsu=color_capture_gray_scale_otsu_threshold(lenna_image)
lenna_image_color_capture_adaptive_gaussian=color_capture_gray_scale_adaptive_gaussian_threshold(lenna_image,15)
show_with_matplotlib(lenna_image_color_capture_otsu,"lenna_image_color_capture_otsu")
show_with_matplotlib(lenna_image_color_capture_adaptive_gaussian,"lenna_image_color_capture_adaptive_gaussian")

"""



"""

lenna_image=bgr_imread("pics\\lenna.png")
lenna_gray_image=bgr2gray(lenna_image)        

lenna_gray_scale_threshold_trans=gray_scale_threshold_trans(lenna_gray_image,100)
lenna_gray_scale_binary_threshold=lenna_gray_scale_threshold_trans.gray_scale_binary_threshold()
lenna_gray_scale_binary_inv_threshold=lenna_gray_scale_threshold_trans.gray_scale_binary_inv_threshold()
lenna_gray_scale_truncation_threshold=lenna_gray_scale_threshold_trans.gray_scale_truncation_threshold()
lenna_gray_scale_tozero_inv_threshold=lenna_gray_scale_threshold_trans.gray_scale_tozero_inv_threshold()
lenna_gray_scale_tozero_threshold=lenna_gray_scale_threshold_trans.gray_scale_tozero_threshold()

show_with_matplotlib(lenna_gray_scale_binary_threshold,"lenna_gray_scale_binary_threshold")
show_with_matplotlib(lenna_gray_scale_binary_inv_threshold,"lenna_gray_scale_binary_inv_threshold")
show_with_matplotlib(lenna_gray_scale_truncation_threshold,"lenna_gray_scale_truncation_threshold")
show_with_matplotlib(lenna_gray_scale_tozero_inv_threshold,"lenna_gray_scale_tozero_inv_threshold")
show_with_matplotlib(lenna_gray_scale_tozero_threshold,"lenna_gray_scale_tozero_threshold")

"""


"""

lenna_image=bgr_imread("pics\\lenna.png")
lenna_gray_image=bgr2gray(lenna_image)
lenna_gray_image_adaptive_trans=gray_scale_adaptive_trans(lenna_gray_image,15)
lenna_gray_image_adaptive_mean,_=lenna_gray_image_adaptive_trans.gray_scale_adaptive_mean_threshold()
lenna_gray_image_adaptive_median,_=lenna_gray_image_adaptive_trans.gray_scale_adaptive_median_threshold()
lenna_gray_image_adaptive_gaussian,_=lenna_gray_image_adaptive_trans.gray_scale_adaptive_gaussian_threshold(6)

show_with_matplotlib(lenna_gray_image_adaptive_mean,"lenna_gray_image_adaptive_mean")
show_with_matplotlib(lenna_gray_image_adaptive_median,"lenna_gray_image_adaptive_median")
show_with_matplotlib(lenna_gray_image_adaptive_gaussian,"lenna_gray_image_adaptive_gaussian")

"""


"""

sudoku_image=bgr_imread("pics\\sudoku.png")
sudoku_gray_image=bgr2gray(sudoku_image)
sudoku_otsu_threshold=gray_scale_otsu_threshold(sudoku_gray_image)
show_with_matplotlib(sudoku_otsu_threshold,"sudoku_gray_scale_otsu_threshold")

sudoku_image=bgr_imread("pics\\sudoku.png")
sudoku_gray_image=bgr2gray(sudoku_image)
sudoku_gray_scale_adaptive_trans=gray_scale_adaptive_trans(sudoku_gray_image,31,2)
sudoku_gray_scale_adaptive_gaussian,_=sudoku_gray_scale_adaptive_trans.gray_scale_adaptive_gaussian_threshold(10)
show_with_matplotlib(sudoku_gray_scale_adaptive_gaussian,"sudoku_gray_scale_adaptive_gaussian")

"""      
    
"""

lenna_image=bgr_imread("pics\\lenna.png")


lenna_color_otsu_threshold_trans=color_otsu_threshold_trans(lenna_image)
lenna_color_binary_otsu_threshold=lenna_color_otsu_threshold_trans.color_binary_otsu_threshold()
lenna_color_binary_inv_otsu_threshold=lenna_color_otsu_threshold_trans.color_binary_inv_otsu_threshold()
lenna_color_truncation_otsu_threshold=lenna_color_otsu_threshold_trans.color_truncation_otsu_threshold()
lenna_color_tozero_inv_otsu_threshold=lenna_color_otsu_threshold_trans.color_tozero_inv_otsu_threshold()
lenna_color_tozero_otsu_threshold=lenna_color_otsu_threshold_trans.color_tozero_otsu_threshold()

show_with_matplotlib(lenna_color_binary_otsu_threshold,"lenna_color_binary_otsu_threshold")
show_with_matplotlib(lenna_color_binary_inv_otsu_threshold,"lenna_color_binary_otsu_threshold")
show_with_matplotlib(lenna_color_truncation_otsu_threshold,"lenna_color_truncation_otsu_threshold")
show_with_matplotlib(lenna_color_tozero_inv_otsu_threshold,"lenna_color_tozero_inv_otsu_threshold")
show_with_matplotlib(lenna_color_tozero_otsu_threshold,"lenna_color_tozero_otsu_threshold")


"""

"""

lenna_image=bgr_imread("pics\\lenna.png")
lenna_color_adaptive_trans=color_adaptive_trans(lenna_image,15)
lenna_color_adaptive_mean,_=lenna_color_adaptive_trans.color_adaptive_mean_threshold()
lenna_color_adaptive_median,_=lenna_color_adaptive_trans.color_adaptive_median_threshold()
lenna_color_adaptive_gaussian,_=lenna_color_adaptive_trans.color_adaptive_gaussian_threshold()

show_with_matplotlib(lenna_color_adaptive_mean,"lenna_color_adaptive_mean")
show_with_matplotlib(lenna_color_adaptive_median,"lenna_color_adaptive_median")
show_with_matplotlib(lenna_color_adaptive_gaussian,"lenna_color_adaptive_gaussian")

"""




############################################################################################## TEST ##############################################################################
