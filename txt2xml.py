# coding=utf-8
import os
import cv2
from xml.dom.minidom import Document


# category_set = ['car']


def custombasename(fullname):
    return os.path.basename(os.path.splitext(fullname)[0])


def limit_value(a, b):
    if a < 1:
        a = 1
    if a >= b:
        a = b - 1
    return a


def readlabeltxt(txtpath, height, width, hbb=False):
    print(txtpath)
    with open(txtpath, 'r') as f_in:  # 打开txt文件
        lines = f_in.readlines()
        splitlines = [x.strip().split(',') for x in lines]  # 根据空格分割
        boxes = []
        for i, splitline in enumerate(splitlines):
            label = splitline[8]
            # if label not in category_set:  # 只书写制定的类别
            #     continue
            x1 = int(float(splitline[0]))
            y1 = int(float(splitline[1]))
            x2 = int(float(splitline[2]))
            y2 = int(float(splitline[3]))
            x3 = int(float(splitline[4]))
            y3 = int(float(splitline[5]))
            x4 = int(float(splitline[6]))
            y4 = int(float(splitline[7]))
            # 如果是hbb
            if hbb:
                xx1 = min(x1, x2, x3, x4)
                xx2 = max(x1, x2, x3, x4)
                yy1 = min(y1, y2, y3, y4)
                yy2 = max(y1, y2, y3, y4)

                xx1 = limit_value(xx1, width)
                xx2 = limit_value(xx2, width)
                yy1 = limit_value(yy1, height)
                yy2 = limit_value(yy2, height)

                box = [xx1, yy1, xx2, yy2, label]
                boxes.append(box)
            else:  # 否则是obb
                x1 = limit_value(x1, width)
                y1 = limit_value(y1, height)
                x2 = limit_value(x2, width)
                y2 = limit_value(y2, height)
                x3 = limit_value(x3, width)
                y3 = limit_value(y3, height)
                x4 = limit_value(x4, width)
                y4 = limit_value(y4, height)

                box = [x1, y1, x2, y2, x3, y3, x4, y4, label]
                boxes.append(box)
    return boxes


def writeXml(tmp, imgname, w, h, d, bboxes, hbb=False):
    doc = Document()
    # owner
    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)

    # source
    source = doc.createElement('source')
    annotation.appendChild(source)
    filename = doc.createElement('filename')
    source.appendChild(filename)
    filename_txt = doc.createTextNode(imgname)
    filename.appendChild(filename_txt)
    origin = doc.createElement('origin')
    source.appendChild(origin)
    origin_txt = doc.createTextNode('GF2/GF3')
    origin.appendChild(origin_txt)

    # research
    research = doc.createElement('research')
    annotation.appendChild(research)

    version = doc.createElement('version')
    research.appendChild(version)
    version_txt = doc.createTextNode("4.0")
    version.appendChild(version_txt)

    provider = doc.createElement('provider')
    research.appendChild(provider)
    provider_txt = doc.createTextNode("Company/School of team")
    provider.appendChild(provider_txt)

    author = doc.createElement('author')
    research.appendChild(author)
    author_txt = doc.createTextNode("team name")
    author.appendChild(author_txt)

    pluginname = doc.createElement('pluginname')
    research.appendChild(pluginname)
    pluginname_txt = doc.createTextNode("Airplane Detection and Recognition")
    pluginname.appendChild(pluginname_txt)

    pluginclass = doc.createElement('pluginclass')
    research.appendChild(pluginclass)
    pluginclass_txt = doc.createTextNode("Detection")
    pluginclass.appendChild(pluginclass_txt)

    time = doc.createElement('time')
    research.appendChild(time)
    time_txt = doc.createTextNode("2020-07-2020-11")
    time.appendChild(time_txt)

    # objects
    objects = doc.createElement('objects')
    annotation.appendChild(objects)

    for bbox in bboxes:
        # threes#
        object_new = doc.createElement("object")
        objects.appendChild(object_new)

        coordinate = doc.createElement('coordinate')
        object_new.appendChild(coordinate)
        coordinate_txt = doc.createTextNode('pixel')
        coordinate.appendChild(coordinate_txt)

        typenode = doc.createElement('type')
        object_new.appendChild(typenode)
        typenode_txt = doc.createTextNode('rectangle')
        typenode.appendChild(typenode_txt)

        description = doc.createElement('description')
        object_new.appendChild(description)
        description_txt = doc.createTextNode('None')
        description.appendChild(description_txt)

        possibleresult = doc.createElement('possibleresult')
        object_new.appendChild(possibleresult)
        name = doc.createElement('name')
        possibleresult.appendChild(name)
        name_txt = doc.createTextNode(str(bbox[-1]))
        name.appendChild(name_txt)
        probability = doc.createElement('probability')
        possibleresult.appendChild(probability)
        probability_txt = doc.createTextNode('1')
        probability.appendChild(probability_txt)

        # threes-1#
        bndbox = doc.createElement('points')
        object_new.appendChild(bndbox)
        for i in range(5):
            point = doc.createElement('point')
            bndbox.appendChild(point)
            point_txt = doc.createTextNode('{:.6f},{:.6f}'.format(bbox[2 * i % 8], bbox[2 * i % 8 + 1]))
            point.appendChild(point_txt)

    xmlname = os.path.splitext(imgname)[0]
    tempfile = os.path.join(tmp, xmlname + '.xml')
    with open(tempfile, 'wb') as f:
        # f.write(doc.toprettyxml(indent='\t', encoding='utf-8'))
        f.write(doc.toprettyxml(indent='\t', encoding='utf-8'))
    return


if __name__ == '__main__':
    data_path = './annotation'
    images_path = os.path.join(data_path, 'images')  # 样本图片路径
    labeltxt_path = os.path.join(data_path, 'txtlabels')  # DOTA标签的所在路径
    anno_new_path = os.path.join(data_path, 'xmllabels_new')  # 新的voc格式存储位置（hbb形式）
    ext = '.tif'  # 样本图片的后缀
    filenames = os.listdir(labeltxt_path)  # 获取每一个txt的名称
    for filename in filenames:
        filepath = labeltxt_path + '/' + filename  # 每一个DOTA标签的具体路径
        picname = os.path.splitext(filename)[0] + ext
        pic_path = os.path.join(images_path, picname)
        im = cv2.imread(pic_path)  # 读取相应的图片
        (H, W, D) = im.shape  # 返回样本的大小
        boxes = readlabeltxt(filepath, H, W, hbb=False)  # 默认是矩形（hbb）得到gt
        if len(boxes) == 0:
            print('文件为空', filepath)
        # 读取对应的样本图片，得到H,W,D用于书写xml

        # 书写xml
        writeXml(anno_new_path, picname, W, H, D, boxes, hbb=False)
        print('正在处理%s' % filename)
