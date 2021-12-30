# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 13:27:58 2021

@author: 邵键帆
"""

"""
Execute some arithmetic transformation in the image 
"""



from matplotlib import pyplot as plt
import numpy as np
from convolution import show_with_matplotlib,show_with_matplotlib_array,image_merge,bgr_imread
from image_resize import array_inter_trans,image_inter_trans




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


######################################################### Image addition and subtraction ################################


#Define a class to add a transforming array to the 3 channels image\
#to make the add or subtract transformation

class add_subtract_array():
    #trans_array: adding or subtracting array
    #original_image: image waiting for transformation
    def __init__(self,trans_array,original_image):
        self.trans_array=trans_array
        self.original_image=original_image
        
    #execute the adding transformation
    def add_array(self):
        #NOTE: cv2.imread() has limit the num in array to [0,255], so we need to converse the data type in \
        #the array to no limit: astype(int)
        #this function is aimed at adding one channel array
        
        trans_channel_list=[]
        for i in range(3):
            trans_channel=self.original_image[:,:,i].astype(int) + self.trans_array
            saturation_trans_channel=saturation_array(trans_channel)
            trans_channel_list.append(saturation_trans_channel)
        trans_image=image_merge(trans_channel_list)
        return trans_image

    #execute the subtracting transformation
    def subtract_array(self):
        #NOTE: cv2.imread() has limit the num in array to [0,255], so we need to converse the data type in \
        #the array to no limit: astype(int)
        #this function is aimed at adding one channel array
        
        trans_channel_list=[]
        for i in range(3):
            trans_channel=self.original_image[:,:,i].astype(int) - self.trans_array
            saturation_trans_channel=saturation_array(trans_channel)
            trans_channel_list.append(saturation_trans_channel)
        trans_image=image_merge(trans_channel_list)
        return trans_image




    

#Define a class to adding a 3-channels array eqaul to a tranforming image 
#to execute the adding or subtracting transformation
class add_subtract_image():
    
    #add_subtract_image: transforming image for adding or subtracting conversion
    #original image: an original image waiting for adding and subtracting transformation
    def __init__(self,add_subtract_image_in,original_image):
        self.add_subtract_image_in=add_subtract_image_in
        self.original_image=original_image
    
    
    #execute the adding transformation to the target image
    def add_image(self):
        #this functionj is aimed at adding 3 channels array
        trans_channel_list=[]
        for i in range(3):
            trans_channel=self.original_image[:,:,i].astype(int)+self.add_subtract_image_in[:,:,i]
            saturation_trans_channel=saturation_array(trans_channel)
            trans_channel_list.append(saturation_trans_channel)
        
        trans_image=image_merge(trans_channel_list)
        return trans_image

    #execute the subtracting transformation to the target image
    def subtract_image(self):
        #this functionj is aimed at adding 3 channels array
        trans_channel_list=[]
        for i in range(3):
            trans_channel=self.original_image[:,:,i].astype(int)-self.add_subtract_image_in[:,:,i]
            saturation_trans_channel=saturation_array(trans_channel)
            trans_channel_list.append(saturation_trans_channel)
        
        trans_image=image_merge(trans_channel_list)
        return trans_image




######################################################### Image addition and subtraction ################################

#################################################################### add weighted ####################################################

"""
def add_weighted_array(array_1,weight_1,array_2,weight_2,intercept,anchor_array):
    if anchor_array == 0:
        
        weighted_trans_array = array_1*weight_1+array_2*weight_2+intercept
        weighted_trans_saturation_trans_array=saturation_array(weighted_trans_array)
    
    return weighted_trans_
    
"""
#################################################################### Bitwise Operation ####################################################

#def a class to execute the array bitwise operation
class array_bitwise():
    
    #initiailize the parameter for the array bitwise class 
    def __init__(self,array_1,array_2,anchor=0):
        #array_1: the first array waiting for bitwise operation
        #array_2: the second array waiting for bitwise operation
        #anchor: the anchor array which will be the standard image size of the transformation
        
        self.array_1=array_1
        self.array_2=array_2
        self.anchor=anchor
    
    

    #return array transformed by the bitwise operatio
    def array_bitwise_and(self):
        
        if self.array_1.shape != self.array_2.shape:
            if self.anchor == 0:
                array_2_trans=array_inter_trans(self.array_2,self.array_1.shape[0],self.array_1.shape[1]).array_inter_bicubic()
                array_1_trans=self.array_1
            else:
                array_1_trans=array_inter_trans(self.array_1,self.array_2.shape[0],self.array_2.shape[1]).array_inter_bicubic()
                array_2_trans=self.array_2
    
    
        trans_array=np.array(array_1_trans & array_2_trans)
        saturation_trans_array=saturation_array(trans_array)
        
        return saturation_trans_array




    #return an array transformed by the bitwise or operation
    def array_bitwise_or(self):
        
        if self.array_1.shape != self.array_2.shape:
            if self.anchor == 0:       
                array_2_trans=array_inter_trans(self.array_2,self.array_1.shape[0],self.array_1.shape[1]).array_inter_bicubic()
                array_1_trans=self.array_1
            else:
                array_1_trans=array_inter_trans(self.array_1,self.array_2.shape[0],self.array_2.shape[1]).array_inter_bicubic()
                array_2_trans=self.array_2
     
        trans_array=array_1_trans | array_2_trans
        saturation_trans_array=saturation_array(trans_array)
        
        return saturation_trans_array       

    #return an array transformed by the bitwise xor operation
    def array_bitwise_xor(self):
        
        if self.array_1.shape != self.array_2.shape:
            if self.anchor == 0:
                array_2_trans=array_inter_trans(self.array_2,self.array_1.shape[0],self.array_1.shape[1]).array_inter_bicubic()
                array_1_trans=self.array_1
            else:
                array_1_trans=array_inter_trans(self.array_1,self.array_2.shape[0],self.array_2.shape[1]).array_inter_bicubic()
                array_2_trans=self.array_2
    
        trans_array=array_1_trans ^ array_2_trans
        saturation_trans_array=saturation_array(trans_array)
        
        return saturation_trans_array



#define an array to execute the bitwise not operation
def array_bitwise_not(array_1):
    
    trans_array=(-np.ones(array_1.shape,dtype="uint8"))-array_1
    saturation_trans_array=saturation_array(trans_array)
    
    return saturation_trans_array


#define a class to execute the bitwise operation against the image 
class image_bitwise():
    #initialize the parameters for the image bitwise class 
    def __init__(self,image_1,image_2,anchor=0):
        #image_1: the first image waiting for for bitwise operation
        #image_2: the second image waiting for bitwise operation
        #anchor: anchor image which will be seemed as the transformed image size
        self.image_1 = image_1
        self.image_2 = image_2
        self.anchor = anchor

    #return an image transformed by bitwise and operation
    def image_bitwise_and(self):
        array_1,array_2=self.image_1[:,:,0],self.image_2[:,:,0]
        if array_1.shape != array_2.shape:
            if self.anchor == 0:
                image_2_trans=image_inter_trans(self.image_2,array_1.shape[0],array_1.shape[1]).image_inter_bicubic()
                image_1_trans=self.image_1
            else:
                image_1_trans=image_inter_trans(self.image_1,array_2.shape[0],array_2.shape[1]).image_inter_bicubic()
                image_2_trans=self.image_2
        
        trans_image=image_1_trans & image_2_trans
        saturation_trans_image=saturation_image(trans_image)
        
        return saturation_trans_image

    #return an image transformed by bitwise or operation
    def image_bitwise_or(self):
        array_1,array_2=self.image_1[:,:,0],self.image_2[:,:,0]
        if array_1.shape != array_2.shape:
            if self.anchor == 0:
                image_2_trans=image_inter_trans(self.image_2,array_1.shape[0],array_1.shape[1]).image_inter_bicubic()
                image_1_trans=self.image_1
            else:
                image_1_trans=image_inter_trans(self.image_1,array_2.shape[0],array_2.shape[1]).image_inter_bicubic()
                image_2_trans=self.image_2
        trans_image=image_1_trans | image_2_trans
        saturation_trans_image=saturation_image(trans_image)
        
        return saturation_trans_image

    #return an image transformed by bitwise xor operation
    def image_bitwise_xor(self):
        array_1,array_2=self.image_1[:,:,0],self.image_2[:,:,0]
        if array_1.shape != array_2.shape:
            if self.anchor == 0:
                image_2_trans=image_inter_trans(self.image_2,array_1.shape[0],array_1.shape[1]).image_inter_bicubic()
                image_1_trans=self.image_1
            else:
                image_1_trans=image_inter_trans(self.image_1,array_2.shape[0],array_2.shape[1]).image_inter_bicubic()
                image_2_trans=self.image_2
        trans_image=image_1_trans ^ image_2_trans
        saturation_trans_image=saturation_image(trans_image)
        
        return saturation_trans_image




#define a function to execute a bitwise not operation against only one image 
def image_bitwise_not(image_1):
    trans_image=(-np.ones(image_1.shape,dtype="uint8"))-image_1
    saturation_trans_image=saturation_image(trans_image)
    
    return saturation_trans_image







#################################################################### Bitwise Operation ####################################################







#################################################################### color2gray(unmature threshold) ####################################################


def array_binary_threshold(array,threshold=int(256/2)):
    trans_array=array.copy()
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if trans_array[i,j]>threshold:
                trans_array[i,j]=1
            else:
                trans_array[i,j]=0
    
    return trans_array

def image_binary_threshold(image,threshold=int(256/2)):
    trans_image=image.copy()
    for channel in range(image.shape[2]):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if trans_image[i,j,channel] > threshold:
                    trans_image[i,j,channel] = 1
                else:
                    trans_image[i,j,channel] = 0
    
    return trans_image




#################################################################### color2gray ####################################################



#################################################################### image_split_merge ####################################################

def image_split(image):
    image_ndim=image.ndim
    channel_list=[]
    for i in range(image_ndim):
        channel_list.append(image[:,:,i])
    return tuple(channel_list)

def image_merge(image_list):
    image_ndim=len(image_list)
    image_merge=np.ones((image_list[0].shape[0],image_list[1].shape[1],3))
    for i in range(image_ndim):
        image_merge[:,:,i] = image_list[i]
    
    return image_merge.astype("uint8")

#################################################################### image_split_merge ####################################################





############################################################################################## TEST ##############################################################################



"""

lenna_image=bgr_imread("pics\\lenna.png")
added_array=np.ones((lenna_image.shape[0],lenna_image.shape[1]),dtype="uint8")*50
added_image_1=bgr_imread("pics/color_spaces.png")
added_resize_image=image_inter_trans(added_image_1,lenna_image.shape[0],lenna_image.shape[1]).image_inter_nn()


lenna_add_subtract_array=add_subtract_array(added_array,lenna_image)
lenna_add_array=lenna_add_subtract_array.add_array()
lenna_subtract_array=lenna_add_subtract_array.subtract_array()

lenna_add_subtract_image=add_subtract_image(added_resize_image,lenna_image)
lenna_add_image=lenna_add_subtract_image.add_image()
lenna_subtract_image=lenna_add_subtract_image.subtract_image()


show_with_matplotlib(added_resize_image,"added_resize_image")
show_with_matplotlib(lenna_add_array,"lenna_add_array")
show_with_matplotlib(lenna_subtract_array,"lenna_subtract_array")
show_with_matplotlib(lenna_add_image,"lenna_add_image")
show_with_matplotlib(lenna_subtract_image,"lenna_subtract_image")

"""




"""

lenna_array=bgr_imread("pics\\lenna.png")[:,:,0]
cat_array=bgr_imread("pics\\color_spaces.png")[:,:,0]
lenna_cat_array_bitwise=array_bitwise(lenna_array,cat_array)
show_with_matplotlib_array(lenna_cat_array_bitwise.array_bitwise_and(),"lenna_color_space_array_bitwise_and")
show_with_matplotlib_array(lenna_cat_array_bitwise.array_bitwise_or(),"lenna_color_space_array_bitwise_or")
show_with_matplotlib_array(lenna_cat_array_bitwise.array_bitwise_xor(),"lenna_color_space_array_bitwise_xor")

"""





"""

lenna_image=bgr_imread("pics\\lenna.png")
cat_image=bgr_imread("pics\\color_spaces.png")


lenna_cat_image_bitwise=image_bitwise(lenna_image,cat_image)
show_with_matplotlib(lenna_cat_image_bitwise.image_bitwise_and(),"lenna_color_spaces_image_bitwise_and")
show_with_matplotlib(lenna_cat_image_bitwise.image_bitwise_or(),"lenna_color_spaces_image_bitwise_or")
show_with_matplotlib(lenna_cat_image_bitwise.image_bitwise_xor(),"lenna_color_spaces_image_bitwise_xor")

"""




############################################################################################## TEST ##############################################################################
