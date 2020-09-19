from __future__ import print_function
import shutil
import os
import cv2
import torch
import argparse
import numpy as np

from models.stela import STELA
from utils.detect import im_detect
from utils.bbox import rbox_2_quad
from utils.utils import is_image, draw_caption

torch.backends.cudnn.enabled = False
from evaluation_v4 import *

import torch.nn as nn
from efficientnet.model import EfficientNet
from torchvision import datasets, models, transforms

classify_size = (224, 224, 3)
class_num = 2
net_name = 'efficientnet-b0'
classes = ['__background__', 'plane']


def crop_image(image, box):
    background = np.zeros(classify_size).astype(np.uint8)
    x0 = max(np.min(box[:, 0]), 0)
    x1 = max(np.max(box[:, 0]), 0)
    y0 = max(np.min(box[:, 1]), 0)
    y1 = max(np.max(box[:, 1]), 0)
    x0, x1, y0, y1 = int(x0), int(x1), int(y0), int(y1)
    patch = image[y0:y1, x0:x1]
    h, w = patch.shape[:2]
    if classify_size[0] / h < classify_size[1] / w:
        new_w = int(classify_size[0] / h * w)
        patch = cv2.resize(patch, (new_w, classify_size[0]))
        patch = np.hstack([np.zeros([classify_size[0], classify_size[1] // 2 - new_w // 2, 3]), patch,
                           np.zeros([classify_size[0],
                                     classify_size[1] - (classify_size[1] // 2 - new_w // 2 + patch.shape[1]), 3])])
    else:
        new_h = int(classify_size[1] / w * h)
        patch = cv2.resize(patch, (classify_size[1], new_h))
        patch = np.vstack([np.zeros([classify_size[0] // 2 - new_h // 2, classify_size[1], 3]), patch,
                           np.zeros([classify_size[0] - (classify_size[0] // 2 - new_h // 2 + patch.shape[0]),
                                     classify_size[1], 3])])
    patch = background + patch
    return patch.astype(np.uint8)


def classify(classifier, patch):
    transform_test = transforms.Compose([
        transforms.Resize(classify_size[:2]),
        transforms.CenterCrop(classify_size[:2]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    patch = torch.from_numpy(patch)
    patch = transform_test(patch)
    size = patch.size()
    patch = patch.reshape(1, size[0], size[1], size[2])
    patch = patch.cuda()
    with torch.no_grad():
        classifier.eval()
        outputs = classifier(patch)
        _, predicted = torch.max(outputs.data, 1)
        pred = predicted.cpu().numpy().tolist()

    return int(pred[0])


def demo(backbone='eb2', weights='weights/deploy_eb_ship_15.pth', ims_dir='sample', target_size=768):
    #
    model = STELA(backbone=backbone, num_classes=2)
    model.load_state_dict(torch.load(weights))
    # model.eval()
    # print(model)

    classifier = EfficientNet.from_name(net_name)
    num_ftrs = classifier._fc.in_features
    classifier._fc = nn.Linear(num_ftrs, class_num)

    classifier = classifier.cuda()
    best_model_wts = 'dataset/weismoke/model/efficientnet-b0.pth'
    classifier.load_state_dict(torch.load(best_model_wts))

    ims_list = [x for x in os.listdir(ims_dir) if is_image(x)]
    import shutil
    shutil.rmtree('output/')
    os.mkdir('output/')
    for _, im_name in enumerate(ims_list):
        im_path = os.path.join(ims_dir, im_name)
        src = cv2.imread(im_path, cv2.IMREAD_COLOR)
        im = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        import time
        # start=time.clock()
        cls_dets = im_detect(model, im, target_sizes=target_size)
        end = time.clock()
        # print('********time*********',end-start)
        # val='/home/jd/projects/haha/chosename/val_plane_split/label_new/'
        """
        if(len(cls_dets)==0):
            print('*********no********',im_name)
            #image_path = os.path.join(img_path, name + ext) #样本图片的名称
            shutil.move(val+im_name[0:-4]+'.txt', 'hard')  #移动该样本图片到blank_img_path
            shutil.move(im_path, 'hard/')     #移动该样本图片的标签到blank_label_path
            continue
        """
        fw = open('output/' + im_name[:-4] + '.txt', 'a')
        fw.truncate()
        for j in range(len(cls_dets)):
            cls, scores = cls_dets[j, 0], cls_dets[j, 1]
            # print ('cls,score',cls,scores)

            bbox = cls_dets[j, 2:]
            # print(bbox)
            if len(bbox) == 4:
                draw_caption(src, bbox, '{:1.3f}'.format(scores))
                cv2.rectangle(src, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(0, 0, 255),
                              thickness=2)
            else:
                pts = np.array([rbox_2_quad(bbox[:5]).reshape((4, 2))], dtype=np.int32)
                # print('####pts####',pts)
                cv2.drawContours(src, pts, 0, color=(0, 255, 0), thickness=2)
                # display original anchors
                # if len(bbox) > 5:
                #     pts = np.array([rbox_2_quad(bbox[5:]).reshape((4, 2))], dtype=np.int32)
                #     cv2.drawContours(src, pts, 0, color=(0, 0, 255), thickness=2)
                patch = crop_image(im, pts)
                pred = classify(classifier, patch)

            fw.write(str(pts.flatten()[0]) + ' ' + str(pts.flatten()[1]) + ' ' + str(pts.flatten()[2]) + ' ' + str(
                pts.flatten()[3]) + ' ' + str(pts.flatten()[4]) + ' ' + str(pts.flatten()[5]) + ' ' + str(
                pts.flatten()[6]) + ' ' + str(pts.flatten()[7]) + ' ' + classes[pred] + '\n')
        fw.close()

        # resize for better shown
        im = cv2.resize(src, (768, 768), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite('output_img/' + im_name, im)

    train_img_dir = '/home/jd/projects/bifpn/sample_plane/'
    groundtruth_txt_dir = '/home/jd/projects/haha/chosename/val_plane_split/label_new/'
    detect_txt_dir = '/home/jd/projects/bifpn/output/'
    Recall, Precision, mAP = compute_acc(train_img_dir, groundtruth_txt_dir, detect_txt_dir)
    print('*******', Recall)
    return Recall, Precision, mAP
    # cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--backbone', type=str, default='eb2')
    parser.add_argument('--weights', type=str, default='./weights/deploy.pth')
    parser.add_argument('--ims_dir', type=str, default='/path/to/yours')
    parser.add_argument('--target_size', type=int, default='800')
    args = parser.parse_args()
    demo(args.backbone, args.weights, args.ims_dir, args.target_size)
