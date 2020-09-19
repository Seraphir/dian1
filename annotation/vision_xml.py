# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 11:13:56 2018

@author: tang
"""

import os, sys  
import glob  
from PIL import Image  
import cv2  
import numpy as np 
import matplotlib
from scipy.spatial import distance as dist
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xml.dom.minidom
def rectangle_box(x1,y1,x2,y2,x3,y3,x4,y4):
	quad = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
	rect = cv2.minAreaRect(quad)
	box1 = cv2.boxPoints(rect)
	box1 = np.int0(box1)

def order_points(pts): #4x2
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]
 
    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
 
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
 
    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0] 
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
 
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order

    coor_tmp = np.array([tl, tr, br, bl], dtype="float32") #4x2
  
    #将距离原点最近的点设为第一个顶点
    [mm, _] = coor_tmp.shape
    dist_min = 0
    min_id = 0  #最小是哪个点
    for jj in range(mm):
        xx, yy = coor_tmp[jj][0], coor_tmp[jj][1]
        distt = xx**2 + yy**2
        if jj == 0 or distt < dist_min:
            dist_min = distt
            min_id = jj
    # print('min_id', min_id)

    coor_last = np.zeros(coor_tmp.shape)
    id_ = 0
    #将距原点最近的那个顶点放在第一个
    for j in range(min_id, mm):
        coor_last[id_, :] = coor_tmp[j, :]
        id_ += 1

    for j in range(min_id):
        coor_last[id_, :] = coor_tmp[j, :]
        id_ += 1


    return coor_last #np.array([tl, tr, br, bl], dtype="float32")




target_img_dir = "/home/jd/projects/haha/dataset/ship/hbbshow/"
  
  
# source train dir  
train_img_dir = "/home/jd/projects/haha/dataset/ship/images/"
train_xml_dir = "/home/jd/projects/haha/dataset/ship/obbxml/"

img_list = []
for file_name in os.listdir(train_img_dir):
    img_list.append(file_name)

f = glob.glob(train_xml_dir + '//*.xml' )
idx=-1
for file in f :
    #print(file)
    idx=idx+1
    #img_name= train_img_dir + img_list[idx]
    idx_P=file.rfind('P',1,len(file)-1)
    idx_D=file.rfind('.',1,len(file)-1)
    name=file[idx_P:idx_D]
    print('name',name)
    #img_name=name+'.xml'
    #img2_name=name+'.tif'
    img_name  = train_img_dir+name+'.tif'
    img2_name = target_img_dir+name+'.tif'
#    print (img_name)
    img = cv2.imread(img_name)

    im = Image.open(img_name)
    imgwidth, imgheight = im.size
    plt.clf()
    plt.imshow(img)
    currentAxis = plt.gca()

    DOMTree = xml.dom.minidom.parse(file)
    annotation = DOMTree.documentElement

    # filename = annotation.getElementsByTagName("filename")[0]
    # # print(filename)
    # imgname = filename.childNodes[0].data + '.tif'
    # print(imgname)

    objects = annotation.getElementsByTagName("object")
    if(not objects):
        continue
    print(file)
    for object in objects:
        
        bbox = object.getElementsByTagName("bndbox")[0]
        x0 = bbox.getElementsByTagName("x1")[0]
        x0 = x0.childNodes[0].data
        x0 = int(float(x0))
#        print(x0)

        y0 = bbox.getElementsByTagName("y1")[0]
        y0 = y0.childNodes[0].data
        y0 = int(float(y0))
#        print(y0)

        x1= bbox.getElementsByTagName("x2")[0]
        x1 = x1.childNodes[0].data
        x1 = int(float(x1))
#        print(x1)

        y1 = bbox.getElementsByTagName("y2")[0]
        y1 = y1.childNodes[0].data
        y1 = int(float(y1))
#        print(y1)

        x2 = bbox.getElementsByTagName("x3")[0]
        x2 = x2.childNodes[0].data
        x2 = int(float(x2))
#        print(x2)

        y2 = bbox.getElementsByTagName("y3")[0]
        y2 = y2.childNodes[0].data
        y2 = int(float(y2))
#        print(y2)

        x3 = bbox.getElementsByTagName("x4")[0]
        x3 = x3.childNodes[0].data
        x3 = int(float(x3))
#        print(x3)

        y3 = bbox.getElementsByTagName("y4")[0]
        y3 = y3.childNodes[0].data
        y3 = int(float(y3))
#        print(y3)

    # x4 = bbox.getElementsByTagName("x4")[0]
    # x4 = x4.childNodes[0].data
    # x4 = float(x4)
    # print(x4)
    #
    # y4 = bbox.getElementsByTagName("y4")[0]
    # y4 = y4.childNodes[0].data
    # y4 = float(y4)
    # print(y4)

        category = object.getElementsByTagName("name")[0]
        category = category.childNodes[0].data
#        print(category)

        quad = np.array([[x0, y0],[x1, y1], [x2, y2], [x3, y3]])
        # quad_x=order_points(quad)
        # rect_box = rectangle_box(x1, y1, x2, y2, x3, y3, x4, y4)
        color_quad = 'r'
        currentAxis.add_patch(plt.Polygon(quad, fill=False, edgecolor=color_quad, linewidth=1))

        plt.text(x0,y0, category, size=2, ha="center", va="center",
                         bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.5, 0.5)))

    plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(imgwidth/100.0/3.0, imgheight/100.0/3.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator()) 
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    plt.margins(0,0) 
    print('img_name2',img2_name)
    plt.savefig(img2_name,dpi=300)
                
