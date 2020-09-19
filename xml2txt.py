# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 11:13:56 2018

@author: tang
"""

import os
import glob
import xml.dom.minidom
from os.path import join as join

# source train dir
xml_dir = "./annotation/xmllabels"
output_dir = "./annotation/txtlabels"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

f = glob.glob(join(xml_dir, '*.xml'))

for idx, filepath in enumerate(f):
    if '\\' in filepath:
        filename = filepath.split('\\')[-1]
    elif '/' in filepath:
        filename = filepath.split('/')[-1]
    else:
        filename = filepath
    print(filepath)
    DOMTree = xml.dom.minidom.parse(filepath)
    annotation = DOMTree.documentElement
    objects = annotation.getElementsByTagName("object")
    output_path = join(output_dir, filename.rstrip('.xml') + '.txt')
    outputfile = open(output_path, 'w+')
    if not objects:
        continue
    for object in objects:
        bbox = object.getElementsByTagName("point")
        for i in range(4):
            point = bbox[i]
            x, y = eval(point.childNodes[0].data)
            print(x, y)
            outputfile.write('{:.0f},{:.0f},'.format(x, y))

        category = object.getElementsByTagName("name")[0]
        category = category.childNodes[0].data
        print(category)
        outputfile.write('{}\n'.format(category))
    outputfile.close()
