# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 12:11:27 2021

@author: 邵键帆
"""

"""
This program is aimed at drawing some contour
"""

import numpy as np
from matplotlib import pyplot as plt
from convolution import show_with_matplotlib,show_with_matplotlib_array
from color_trans import bgr2gray,gray2bgr_show
from contour_detection import image_contour_detection





############################################################################################## Drawing basic shape ##############################################################################

#return the point list
def point_list(x_center,y_center,radius):
    circle_list_1=[]
    x_min,x_max=x_center-radius,x_center+radius
    y_min,y_max=y_center-radius,y_center+radius
    for x in range(x_min,x_max+1):
        for y in range(y_min,y_max+1):
            if (x-x_center)**2+(y-y_center)**2 <= radius**2:
                circle_list_1.append([x,y])

    return circle_list_1

#return the circle list
def circle_list(x_center,y_center,radius,size):
    circle_bigger_list=point_list(x_center,y_center,radius)
    circle_small_list=point_list(x_center,y_center,radius-size)
    circle_list=[point for point in circle_bigger_list if point not in circle_small_list]
    
    return circle_list

#drawing the point 
def draw_point(location_array,image,color,size):
    trans_image=image.copy()
    location_list=location_array.tolist()
    for center_point in location_list:
        for circle_point in point_list(center_point[0],center_point[1],size):
            trans_image[circle_point[0],circle_point[1],:]=np.array(color)
    return trans_image


#drawing the circle 
def draw_circle(location_array,image,color,radius,size):
    trans_image=image.copy()
    location_list=location_array.tolist()
    for center_point in location_list:
        for circle_point in circle_list(center_point[0],center_point[1],radius,size):
            trans_image[circle_point[0],circle_point[1],:]=np.array(color)
    return trans_image    




def line_point(point_1,point_2):
    x1,y1=point_1[0],point_1[1]
    x2,y2=point_2[0],point_2[1]
    x_max,x_min=max([x1,x2]),min([x1,x2])
    y_max,y_min=max([y1,y2]),min([y1,y2])
    point_list=[]
    for x in range(x_min,x_max+1):
        y=(((y1-y2)/(x1-x2))*(x-x2))+y2
        point_list.append([x,y])
    return point_list

def line_extend_point(point_1,point_2,size):
    line_point_exact=line_point(point_1,point_2)
    line_extend_point_list=[]
    for point in line_point_exact:
        for i in range(-size,size+1):
            line_extend_point_list.append([point[0],int(point[1]+i)])
    
    return line_extend_point_list

#drawing the line
def draw_line(location_array,image,color,size):
    tran_image=image.copy()
    location_list=location_array.tolist()
    location_list.append(location_list[0])
    location_list_len=len(location_list)
    point_pair_list=[]
    points_list=[]
    for i in range(location_list_len-1):
        point_pair_list.append([location_list[i],location_list[i+1]])
    for point_pair in point_pair_list:
        line_extend_point_list_1=line_extend_point(point_pair[0],point_pair[1],size)
        points_list.extend(line_extend_point_list_1)
    for point in points_list:
        tran_image[point[0],point[1],:]=color
    return tran_image





#drawing the rectangle 
def draw_rectangle(location_array,image,color):
    tran_image=image.copy()
    location_list=location_array.tolist()
    point_1,point_2=location_list[0],location_list[1]
    x1,y1=point_1[0],point_1[1]
    x2,y2=point_2[0],point_2[1]
    x_max,x_min=max([x1,x2]),min([x1,x2])
    y_max,y_min=max([y1,y2]),min([y1,y2])
    for x in range(x_min,x_max+1):
        for y in range(y_min,y_max+1):
            tran_image[x,y,:]=color
    return tran_image




############################################################################################## Drawing basic shape ##############################################################################



############################################################################################## TEST ##############################################################################

"""

background=np.zeros((640,640,3),dtype="uint8")
circle_location=np.array([[200,200]])
circle=draw_circle(circle_location,background,[255,0,255],100,50)
show_with_matplotlib(circle,"circle")
circle_contour=image_contour_detection(circle)
show_with_matplotlib(gray2bgr_show(circle_contour),"circle_contour")

"""

"""

cnts=np.array([[[600,320]],
                   [[563,460]],
                   [[460,562]],
                   [[320,600]],
                   [[180,563]],
                   [[78,460]],
                   [[40,320]],
                   [[77,180]],
                   [[179,78]],
                   [[319,40]],
                   [[459,77]],
                   [[562,179]],
                   ],dtype="uint32")
point_location=np.squeeze(cnts)
background=np.zeros((640,640,3),dtype="uint8")
circle_1=draw_point(point_location,background,[255,0,255],5)
show_with_matplotlib(circle_1,"multiple_point")

line_tran_image=draw_line(point_location,background,[0,255,255],9)
line_point_tran_image=draw_point(point_location,line_tran_image,[255,0,255],20)
show_with_matplotlib(line_point_tran_image,"line_point_tran_image")

rectangle_location=np.array([[20,30],
                             [40,80],
                             [600,130]])
rectangle_image=draw_line(rectangle_location,background,[0,255,255],3)
show_with_matplotlib(rectangle_image,"rectangle_image")

"""








############################################################################################## TEST ##############################################################################


        
            
            
    