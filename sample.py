import cv2
import os
import numpy as np
from vizer.draw import draw_boxes

dataset_dir = "/Users/seraph/Documents/Datasets/crop"
image_dir = os.path.join(dataset_dir, "images")
label_dir = os.path.join(dataset_dir, "labels")
output_dir = "./out"
output_size = (224, 224, 3)  # (h,w)


def load_label(filepath):
    f = open(filepath, 'r')
    lines = f.readlines()
    boxes = []
    names = []
    for line in lines:
        box = np.array([eval(x) for x in line.split(',')[:-1]])
        name = line.split(',')[-1]
        name = name.rstrip('\n')
        box = np.reshape(box, (4, 2))
        # box = np.flip(box, axis=1)
        boxes.append(box)
        names.append(name)
    boxes = np.array(boxes).astype(np.int32)
    return boxes, names


def crop_image(image, box):
    background = np.zeros(output_size).astype(np.uint8)
    x0 = max(np.min(box[:, 0]), 0)
    x1 = max(np.max(box[:, 0]), 0)
    y0 = max(np.min(box[:, 1]), 0)
    y1 = max(np.max(box[:, 1]), 0)
    patch = image[y0:y1, x0:x1]
    h, w = patch.shape[:2]
    if output_size[0] / h < output_size[1] / w:
        new_w = int(output_size[0] / h * w)
        patch = cv2.resize(patch, (new_w, output_size[0]))
        patch = np.hstack([np.zeros([output_size[0], output_size[1] // 2 - new_w // 2, 3]), patch,
                           np.zeros([output_size[0],
                                     output_size[1] - (output_size[1] // 2 - new_w // 2 + patch.shape[1]), 3])])
    else:
        new_h = int(output_size[1] / w * h)
        patch = cv2.resize(patch, (output_size[1], new_h))
        patch = np.vstack([np.zeros([output_size[0] // 2 - new_h // 2, output_size[1], 3]), patch,
                           np.zeros([output_size[0] - (output_size[0] // 2 - new_h // 2 + patch.shape[0]),
                                     output_size[1], 3])])
    patch = background + patch
    return patch.astype(np.uint8)


image_ls = os.listdir(image_dir)
image_boxes_names_ls = []
for image_name in image_ls:
    if image_name.split('.')[-1] in ["png", "jpg", "bmp"]:
        index = image_name.split('.')[0]
        image = cv2.imread(os.path.join(image_dir, image_name))
        image = cv2.resize(image, dsize=(0, 0), fx=0.2, fy=0.2)
        print(image.shape)
        boxes, names = load_label(os.path.join(label_dir, index + '.txt'))
        image_boxes_names_ls.append([image, boxes, names, index])

for image, boxes, names, index in image_boxes_names_ls:
    for i, box in enumerate(boxes):
        name = names[i]
        target = crop_image(image, box)
        # image = cv2.polylines(image, np.int32([box]), True, color=(0, 0, 255), thickness=10)
        if not os.path.exists(os.path.join(output_dir, name)):
            os.makedirs(os.path.join(output_dir, name))
        save_path = os.path.join(output_dir, name, index + '_{:02d}.png'.format(i))
        cv2.imwrite(save_path, target)
        # cv2.imshow(name, target)
        # cv2.waitKey(0)
