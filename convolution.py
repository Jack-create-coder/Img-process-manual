# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 21:28:22 2021

@author: 邵键帆
"""

"""
convolution(Not mutual relevance)

"""

#import the relevant package
import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib import image as img







################################################################# convolution transformation(one channel) ####################################################



#split the image
def image_split(image):
    image_ndim=image.ndim
    channel_list=[]
    for i in range(image_ndim):
        channel_list.append(image[:,:,i])
    return tuple(channel_list)

#merge the image
def image_merge(image_list):
    image_ndim=len(image_list)
    image_merge=np.ones((image_list[0].shape[0],image_list[0].shape[1],3),dtype="uint8")
    for i in range(image_ndim):
        image_merge[:,:,i] = image_list[i]
    
    return image_merge.astype("uint8")


#implement the saturation transformation
def saturation_array(array,up_threshold_value=255,low_threshold_value=0):
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i,j] > up_threshold_value:
                array[i,j] = 255
            elif array[i,j] < low_threshold_value:
                array[i,j] = 0
            
    return array


#define a class to carry out relevant convolution trans
#This class is serving for one channel image 
#Here convolution have not execute saturation transformation\
#If you need to execute the saturation transformation, Please\
#execute the proceding procedure
#The range of imported array have been loose(no limit)


class convolution_trans:
    
    #initial the relevant parameters
    def __init__(self,array,kernel,s,t):
        #kernel: imported kernel for image transformation
        #array: one channel array of image array(e.g b_array/g_array/r_array)
        #s: sliding step of row dimension 
        #t: sliding step of column dimension
        
        self.kernel=kernel
        self.array=array.astype(int)
        self.s=s
        self.t=t
        
    
    #padding zero or specific num to the original array
    def pad_zero(self,row_num=0,column_num=0):
        #row:(n,m) --- forefront padding n * 0 and backmost padding m * 0
        #colmn: (i,j) --- left side padding i * 0 and right side padding j * 0
        row,column=self.pad_num()
        padding_array=np.pad(self.array,(row,column),"constant",constant_values=(row_num,column_num))
        return padding_array
    
    
    #return to row and col padding num 
    def pad_num(self):
        #m,n refer to the array's row nums and column nums
        #k,j refer to the kernel's row nums and column nums
        #p refer padding zero num for row 
        #q refer padding zero num for column
        
        m,n=self.array.shape[0],self.array.shape[1]
        k,j=self.kernel.shape[0],self.kernel.shape[1]
        p=((self.s-1)*m-self.s+k)/2
        q=((self.t-1)*n-self.t+j)/2
        
        if p % 1 == 0 and q % 1 == 0:
            return (int(p),int(p)),(int(q),int(q))
        
        else:
            print("Error: Wrong number of the input kernel size or step.")    
    
    #this function execute the convolution transformation
    def convolution(self,row_num=0,column_num=0):
        #NOTE: make sure that the kernel size of row and column are both odd
        padding_array=self.pad_zero(row_num,column_num)
        starting_row=int((self.kernel.shape[0]+1)/2)
        starting_col=int((self.kernel.shape[1]+1)/2)
        extend_height=int((self.kernel.shape[0]-1)/2)
        extend_width=int((self.kernel.shape[1]-1)/2)
        array_row=padding_array.shape[0]
        array_col=padding_array.shape[1]
        image_list=[]
        for i in range(starting_row-1,array_row-starting_row+1,self.s):
            line=[]
            for j in range(starting_col-1,array_col-starting_col+1,self.t):
                corresponding_array=padding_array[i-extend_height:i+extend_height+1,j-extend_width:j+extend_width+1]
                line.append(np.sum(np.multiply(self.kernel,corresponding_array)))
            
            image_list.append(line)
        return np.array(image_list).astype(int)

################################################################# convolution transformation(one channel) ####################################################




################################################################# convolution extra transformation based on the one channel convolution transformation ####################################################


#define a function aimed at the image(three channels: b, g, r) to make the convolution transformation
def image_convolution_trans(image,kernel,s,t):
    #image: imported image
    #kernel: kernel for implementation of convolutional transformation
    #s: sliding step of row dimension
    #t: sliding step of column dimension
    
    b,g,r = image_split(image)
    b_trans=convolution_trans(b,kernel,s,t)
    g_trans=convolution_trans(g,kernel,s,t)
    r_trans=convolution_trans(r,kernel,s,t)
    
    b_channel=saturation_array(b_trans.convolution())
    g_channel=saturation_array(g_trans.convolution())
    r_channel=saturation_array(r_trans.convolution())
    trans_image=image_merge([b_channel,g_channel,r_channel])
    
    return trans_image


"""
def convolution_weighted_trans(weight_1,kernel_1,weight_2,kernel_2):
"""  



################################################################# convolution extra transformation based on the one channel convolution transformation ####################################################


####################################################################################### image visualization tools #######################################################################


#define a functionj amied at show the image in matplotlib form(r,g,b) \ 
#with the image title
def show_with_matplotlib(bgr_image,title):
    rgb_image=bgr_image[:,:,::-1]
    plt.imshow(rgb_image)
    plt.title(title)
    plt.show()
    


#define a functionj amied at show the one channel image \ 
#with the image title
def show_with_matplotlib_array(array,title):
    plt.imshow(array)
    plt.title(title)
    plt.show()
    

    
def show_with_matplotlib_subplot(image,title,pos,row_num,col_num):
    rbg_image=image[:,:,::-1]
    ax=plt.subplot(row_num,col_num,pos)
    plt.imshow(rbg_image)
    plt.title(title)

def bgr_imread(img_path):
    
    bgr_image=img.imread(img_path)
    if bgr_image.shape[2]==4:
        bgr_image=np.delete(bgr_image,3,axis=2)
    bgr_image=bgr_image[:,:,::-1]
    bgr_image_trans=bgr_image*255
    bgr_image_trans_1=bgr_image_trans.astype("uint8")
    return bgr_image_trans_1

####################################################################################### image visualization tools #######################################################################
            
    

    

############################################################################################## classic filter ###########################################################################


"""

1.laplace filter

laplace_filter=np.array([[0,1,0],
                         [1,-4,1],
                         [0,1,0]])

2.laplace sharpen filter

laplace_sharpen_filter=np.array([[0,-1,0],
                                 [-1,5,-1],
                                 [0,-1,0]])




3.sobel sharpen filter

(1) sobel gradient_x filter 

sobel_gx_filter=np.array([[-1,0,1],
                          [-2,0,2],
                          [-1,0,1]])

(2) sobel gradient_y filter

sobel_gy_filter=np.array([[1,2,1],
                          [0,0,0],
                          [-1,-2,-1]])



"""



############################################################################################## classic filter ###########################################################################







############################################################################################## TEST ##############################################################################

"""

laplace_filter=np.array([[0,1,0],
                         [1,-4,1],
                         [0,1,0]])
laplace_sharpen_filter=np.array([[0,-1,0],
                                 [-1,5,-1],
                                 [0,-1,0]])
sobel_gx_filter=np.array([[-1,0,1],
                          [-2,0,2],
                          [-1,0,1]])

sobel_gy_filter=np.array([[1,2,1],
                          [0,0,0],
                          [-1,-2,-1]])


lenna_image=bgr_imread("pics/lenna.png")



lenna_laplace_image=image_convolution_trans(lenna_image,laplace_filter,1,1)
lenna_laplace_sharpen_image=image_convolution_trans(lenna_image,laplace_sharpen_filter,1,1)
lenna_sobel_gx_sharpen_image=image_convolution_trans(lenna_image,sobel_gx_filter,1,1)
lenna_sobel_gy_sharpen_image=image_convolution_trans(lenna_image,sobel_gy_filter,1,1)

show_with_matplotlib(lenna_image,"lenna_original_image")
show_with_matplotlib(lenna_laplace_image,"lenna_laplace_image")
show_with_matplotlib(lenna_laplace_sharpen_image,"lenna_laplace_sharpen_image")
show_with_matplotlib(lenna_sobel_gx_sharpen_image,"lenna_sobel_gx_sharpen_image")
show_with_matplotlib(lenna_sobel_gy_sharpen_image,"lenna_sobel_gy_sharpen_image")

"""

############################################################################################## TEST ##############################################################################
