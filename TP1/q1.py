import numpy as np
import json
import os
import cv2

gt_path = 'data/part1/gt.json'
images_path = 'data/part1/images'

# images_names = [x.split('.')[0] for x in os.listdir(images_path)]
images_names = os.listdir(images_path)

with open(gt_path) as f:
    gt = json.load(f)

annotations = gt['annotations']
categories = gt['categories']

for image_name in images_names:
    img = cv2.imread(os.path.join(images_path, image_name), 1)

    name = image_name.split('.')[0]

    for ann in annotations:
        if ann['image'] == name:
            x, y, w, h = np.array(ann['bbox']).astype('int32')
            if ann['category_id'] == 8:
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), -1)
            else:
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.imshow(name, img)
    cv2.waitKey(0)
    save_name = name + '_result.jpg'
    cv2.imwrite(os.path.join(images_path, save_name), img)
