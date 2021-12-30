# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 02:10:00 2021

@author: 邵键帆
"""

"""

This Program is aimed at coding the bilateral filter 
Note: It's better to limit the sigma_space and sigma_color to [0,1]
      It's better to limit the col and row of convolution kernel are \
      both odd
Standard form: xxx_trans(one channel trans)/ image_xxx_trans(3 channels trans)

"""

#import the relevant package
import numpy as np
import math

from matplotlib import pyplot as plt
from convolution import image_convolution_trans,show_with_matplotlib,convolution_trans,bgr_imread,image_merge
from arithmetic_operation import saturation_array


################################################################### Normal(Average/median/max/minimum)transformation ###############################################


#define the class to execute the Normal filtering transformation
#only for the one channel array(for specific channel for three channels pic)
class normal_trans():
    
    #initialize the relevant parameter for the normal trans
    def __init__(self,array,kernel_row,kernel_col,s,t):
        #array: specific channel of the image 
        #kernel_row: number of the kernel row
        #kernel_col: number of the kernel column
        #s: sliding step of the row dimension
        #t: sliding step of the column dimension
        self.array=array.astype(int)
        self.kernel_row=kernel_row
        self.kernel_col=kernel_col
        self.s=s
        self.t=t
    
    #return the averaging filtering array
    def average_trans(self):
        kernel_array=(1/(self.kernel_row*self.kernel_col))*np.ones((self.kernel_row,self.kernel_col))
        average_trans=convolution_trans(self.array,kernel_array,self.s,self.t)
        average_trans_array=average_trans.convolution()
        
        return average_trans_array

    #return the median filtering array
    def median_trans(self):
        fake_kernel=np.ones((self.kernel_row,self.kernel_col))
        padding_array=convolution_trans(self.array,fake_kernel,self.s,self.t).pad_zero()
        starting_row,starting_col=int((self.kernel_row-1)/2),int((self.kernel_col-1)/2)
        padding_array_row,padding_array_col=padding_array.shape[0],padding_array.shape[1]
        extend_height=int((self.kernel_row-1)/2)
        extend_width=int((self.kernel_col-1)/2)
        trans_list=[]
        for i in range(starting_row,padding_array_row-starting_row,self.s):
            line_list=[]
            for j in range(starting_col,padding_array_col-starting_col,self.t):
                corresponding_array=padding_array[i-extend_height:i+extend_height+1,j-extend_width:j+extend_width+1]
                median_value=np.median(corresponding_array)
                line_list.append(median_value)
            
            trans_list.append(line_list)
        return np.array(trans_list).astype(int)

    #return the maximum filtering array            
    def max_trans(self):
        fake_kernel=np.ones((self.kernel_row,self.kernel_col))
        padding_array=convolution_trans(self.array,fake_kernel,self.s,self.t).pad_zero()
        starting_row,starting_col=int((self.kernel_row-1)/2),int((self.kernel_col-1)/2)
        padding_array_row,padding_array_col=padding_array.shape[0],padding_array.shape[1]
        extend_height=int((self.kernel_row-1)/2)
        extend_width=int((self.kernel_col-1)/2)
        trans_list=[]
        for i in range(starting_row,padding_array_row-starting_row,self.s):
            line_list=[]
            for j in range(starting_col,padding_array_col-starting_col,self.t):
                corresponding_array=padding_array[i-extend_height:i+extend_height+1,j-extend_width:j+extend_width+1]
                max_value=corresponding_array.max()
                line_list.append(max_value)
            
            trans_list.append(line_list)
        return np.array(trans_list).astype(int)            
    
    #return the minimum filtering array
    def min_trans(self):
        fake_kernel=np.ones((self.kernel_row,self.kernel_col))
        padding_array=convolution_trans(self.array,fake_kernel,self.s,self.t).pad_zero()
        starting_row,starting_col=int((self.kernel_row-1)/2),int((self.kernel_col-1)/2)
        padding_array_row,padding_array_col=padding_array.shape[0],padding_array.shape[1]
        extend_height=int((self.kernel_row-1)/2)
        extend_width=int((self.kernel_col-1)/2)
        trans_list=[]
        for i in range(starting_row,padding_array_row-starting_row,self.s):
            line_list=[]
            for j in range(starting_col,padding_array_col-starting_col,self.t):
                corresponding_array=padding_array[i-extend_height:i+extend_height+1,j-extend_width:j+extend_width+1]
                min_value=corresponding_array.min()
                line_list.append(min_value)
            
            trans_list.append(line_list)
        return np.array(trans_list).astype(int)




#define the class to execute the filtering transformation for\
#the image (3 channels)    
class image_normal_trans():
    
    #initialize the relevant package for the filtering transformation
    def __init__(self,image_array,kernel_row,kernel_col,s,t):
        #image_array: imported image (three channels)
        #kernel_row: the number of the kernel rows
        #kernel_col: the number of the kernel columns
        #s: sliding step of the row dimension
        #t: sliding step of the column dimension
        self.image_array=image_array.astype(int)
        self.kernel_row=kernel_row
        self.kernel_col=kernel_col
        self.s=s
        self.t=t

    #return average filtering image 
    def image_average_trans(self):
        kernel_array=(1/(self.kernel_row*self.kernel_col))*np.ones((self.kernel_row,self.kernel_col))
        image_average_array=image_convolution_trans(self.image_array,kernel_array,self.s,self.t)
        return image_average_array
    
    #return the normal_trans class for each channel 
    def normal_trans(self):
        b,g,r=self.image_array[:,:,0],self.image_array[:,:,1],self.image_array[:,:,2]
        b_trans=normal_trans(b,self.kernel_row,self.kernel_col,self.s,self.t)
        g_trans=normal_trans(g,self.kernel_row,self.kernel_col,self.s,self.t)
        r_trans=normal_trans(r,self.kernel_row,self.kernel_col,self.s,self.t)
        return b_trans,g_trans,r_trans
    
    #return median filtering image
    def image_median_trans(self):
        b_trans,g_trans,r_trans=self.normal_trans()
        b_median=b_trans.median_trans()
        g_median=g_trans.median_trans()
        r_median=r_trans.median_trans()
        image_median_trans=image_merge([b_median,g_median,r_median])
        return image_median_trans
    
    #return max filtering image 
    def image_max_trans(self):
        b_trans,g_trans,r_trans=self.normal_trans()
        b_max=b_trans.max_trans()
        g_max=g_trans.max_trans()
        r_max=r_trans.max_trans()
        image_max_trans=image_merge([b_max,g_max,r_max])
        return image_max_trans

    #return minimum filtering image 
    def image_min_trans(self):
        b_trans,g_trans,r_trans=self.normal_trans()
        b_min=b_trans.min_trans()
        g_min=g_trans.min_trans()
        r_min=r_trans.min_trans()
        image_min_trans=image_merge([b_min,g_min,r_min])
        return image_min_trans





################################################################### Normal(Average/median/maximum/minimum)transformation ###############################################




################################################################### Gaussian transformation ###############################################


#define a class to create the gaussian kernel
class gaussian_kernel():
    
    #intialize the relevant parameter for the gaussian kernel transformation
    def __init__(self,row_num,col_num,sigma_space):
        #row_num: the number of the row in gaussian kernel
        #col_num: the number of the column in gaussian kernel
        #sigma_space: the standard deviation of the gaussian kernel 
        self.row_num=row_num
        self.col_num=col_num
        self.sigma_space=sigma_space
    

    #return gaussian kernel in floating form
    def gaussian_kernel_float(self):
        #make sure that the row_num and col_num are both odd
        
        center_row=(self.row_num-1)/2
        center_col=(self.col_num-1)/2
        
        kernel_list=[]
        for i in range(self.row_num):
            line_list=[]
            for j in range(self.col_num):
                gaussian_value=(1/(2*math.pi*(self.sigma_space**2)))*math.exp(-(((i-center_row)**2+(j-center_col)**2)/(2*(self.sigma_space**2))))
                line_list.append(gaussian_value)
            kernel_list.append(line_list)
        return np.array(kernel_list)




    #return the gaussian kernel in int form
    def gaussian_kernel_int(self):
        #make sure that the row_num and col_num are both odd
        
        center_row=(self.row_num-1)/2
        center_col=(self.col_num-1)/2
        
        kernel_list=[]
        for i in range(self.row_num):
            line_list=[]
            for j in range(self.col_num):
                gaussian_value=(1/(2*math.pi*(self.sigma_space**2)))*math.exp(-(((i-center_row)**2+(j-center_col)**2)/(2*(self.sigma_space**2))))
                line_list.append(gaussian_value)
            kernel_list.append(line_list)
        kernel_array=np.array(kernel_list)
        kernel_array_int=kernel_array/(kernel_array[0,0])
        return kernel_array_int



#return an array trnasformed by gaussian filter
def gaussian_trans(array,row_num,col_num,sigma_space,s,t):
    gaussian_kernel_trans=gaussian_kernel(row_num,col_num,sigma_space)
    gaussian_float_kernel=gaussian_kernel_trans.gaussian_kernel_float()
    array_gaussian_trans=convolution_trans(array,gaussian_float_kernel,s,t).convolution()
    return saturation_array(array_gaussian_trans)



################################################################### Gaussian transformation ###############################################






################################################################### bilateral transformation ###############################################



#define a class to create the bilateral image
class bilateral_trans():
    
    #intialize the initial parameter for the 
    def __init__(self,array,row_num,col_num,sigma_space,s,t):
        #array: one channel image array
        #row_num: the number of rows in gaussian kernel
        #col_num: the number of columns in gaussian kernel 
        #sigma_space: the standard deviation of the gaussian kernel
        #s: the sliding step of the row dimension
        #t: the sliding step of the column dimension
        
        self.row_num=row_num
        self.col_num=col_num
        self.sigma_space=sigma_space
        self.array=array.astype(int)
        self.s=s
        self.t=t
    
    #return a specific gaussian kernel for the bilateral filter
    def gaussian_kernel_float_bilateral(self):
        #make sure that the row_num and col_num are both odd
        
        center_row=(self.row_num-1)/2
        center_col=(self.col_num-1)/2
        
        kernel_list=[]
        for i in range(self.row_num):
            line_list=[]
            for j in range(self.col_num):
                gaussian_value=math.exp(-(((i-center_row)**2+(j-center_col)**2)/(2*(self.sigma_space**2))))
                line_list.append(gaussian_value)
            kernel_list.append(line_list)
        return np.array(kernel_list)

    #padding zero or other specific num for the original array
    def pad_zero(self,row_num=0,column_num=0):
        #row:(n,m) --- forefront padding n * 0 and backmost padding m * 0
        #colmn: (i,j) --- left side padding i * 0 and right side padding j * 0
        #row_num: the padding num of row is zero, the default number can be modified 
        #column_num: the padding num of column is zero, 
        row,column=self.pad_num()
        padding_array=np.pad(self.array,(row,column),"constant",constant_values=(row_num,column_num))
        return padding_array

    #return the row of col padding num 
    def pad_num(self):
        #m,n refer to the array's row nums and column nums
        #k,j refer to the kernel's row nums and column nums
        #p refer padding zero num for row 
        #q refer padding zero num for column
        kernel=self.gaussian_kernel_float_bilateral()
        m,n=self.array.shape[0],self.array.shape[1]
        k,j=kernel.shape[0],kernel.shape[1]
        p=((self.s-1)*m-self.s+k)/2
        q=((self.t-1)*n-self.t+j)/2
        
        if p % 1 == 0 and q % 1 == 0:
            return (int(p),int(p)),(int(q),int(q))
        
        else:
            print("Error: Wrong number of the input kernel size or step.")
        
    #execute the bilateral transformation against the import array
    def bilateral_convolution(self,sigma_color,row_num=0,column_num=0):
        #NOTE: make sure that the kernel size of row and column are both odd
        #sigma_
        kernel=self.gaussian_kernel_float_bilateral()
        padding_array=self.pad_zero(row_num,column_num)
        starting_row=int((kernel.shape[0]+1)/2)
        starting_col=int((kernel.shape[1]+1)/2)
        extend_height=int((kernel.shape[0]-1)/2)
        extend_width=int((kernel.shape[1]-1)/2)
        array_row=padding_array.shape[0]
        array_col=padding_array.shape[1]
        image_list=[]
        for i in range(starting_row-1,array_row-starting_row+1,self.s):
            line=[]
            for j in range(starting_col-1,array_col-starting_col+1,self.t):
                corresponding_array=padding_array[i-extend_height:i+extend_height+1,j-extend_width:j+extend_width+1]
                
                corresponding_row,corresponding_col=corresponding_array.shape[0],corresponding_array.shape[1]
                
                center_row,center_col=int((corresponding_row-1)/2),int((corresponding_col-1)/2)
                
                corr_list=[]
                for p in range(corresponding_row):
                    corr_line=[]
                    for q in range(corresponding_col):
                        
                        corr_value=math.exp(-(((corresponding_array[p,q]-corresponding_array[center_row,center_col]))**2)/(2*(sigma_color**2)))
                        
                        corr_line.append(corr_value)
                    
                    corr_list.append(corr_line)
                
                color_array=np.array(corr_list)
                
                line.append(np.sum(np.multiply(np.multiply(kernel,color_array),corresponding_array)))
            image_list.append(line)
        
        return saturation_array(np.array(image_list).astype(int))


#define a function to execute the bilateral transformation aganist the import image\
#(3 channels: b,g,r)
def image_bilateral_trans(image,row_num,col_num,sigma_space,sigma_color,s,t):
    #image: imported image(b,g,r)
    #row_num: kernel row num
    #col_num: kernel col num
    #sigma_space: the standard deviation relate to location(x and y dimension are the same)
    #sigma_color: the standard deviation related to gap of neighbor color
    (b,g,r) = image[:,:,0],image[:,:,1],image[:,:,2]
    b_trans=bilateral_trans(b,row_num,col_num,sigma_space,s,t)
    g_trans=bilateral_trans(g,row_num,col_num,sigma_space,s,t)
    r_trans=bilateral_trans(r,row_num,col_num,sigma_space,s,t)
    
    b_channel=b_trans.bilateral_convolution(sigma_color)
    g_channel=g_trans.bilateral_convolution(sigma_color)
    r_channel=r_trans.bilateral_convolution(sigma_color)
    
    trans_image=image_merge([b_channel,g_channel,r_channel])
    
    return trans_image




################################################################### bilateral transformation ###############################################




############################################################################################## TEST ##############################################################################



"""

lenna_image=bgr_imread("pics\\lenna.png")
lenna_trans=image_normal_trans(lenna_image,3,3,1,1)
lenna_average_trans=lenna_trans.image_average_trans()
lenna_median_trans=lenna_trans.image_median_trans()
lenna_max_trans=lenna_trans.image_max_trans()
lenna_min_trans=lenna_trans.image_min_trans()

show_with_matplotlib(lenna_average_trans,"lenna_average_trans")
show_with_matplotlib(lenna_median_trans,"lenna_median_trans")
show_with_matplotlib(lenna_max_trans,"lenna_max_trans")
show_with_matplotlib(lenna_min_trans,"lenna_min_trans")

"""

"""

lenna_image=bgr_imread("pics\\lenna.png")
lenna_bilateral=image_bilateral_trans(lenna_image,3,3,0.3,6,1,1)

show_with_matplotlib(lenna_bilateral,"lenna_bilateral")

"""

############################################################################################## TEST ##############################################################################
