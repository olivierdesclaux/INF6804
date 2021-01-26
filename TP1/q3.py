import numpy as np
import json
import os
import cv2
import matplotlib.pyplot as plt
gt_path = 'data/part1/gt.json'
images_path = 'data/part1/images'
IF_HSV = False



# We extract the data from our json
with open(gt_path) as f:
    gt = json.load(f)
annotations = gt['annotations']
categories = gt['categories']
skates = ['skate1', 'skate2']

k=1 # for graphic purposes only
plt.figure(figsize = (15, 10))


for skate in skates:
    image = cv2.imread(os.path.join(images_path, skate + '.jpg'), 1)
    if IF_HSV:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sub_annotations = [ann for ann in annotations if ann['image'] == skate] #we get only the appropriate skate annotations

    #Histogram settings
    histSize = 256 # number of bins
    histRange = [0, 256]
    colors = ['b', 'g', 'r']
    for ann in sub_annotations:
        if ann['category_id'] == 1: #category_id of a person
            #extract roi
            x, y, w, h = np.array(ann['bbox']).astype('int32')
            person = image[y: y+h, x: x+w]
            
            # we show the person
            plt.subplot(2, 2, k)
            plt.axis('off')
            if IF_HSV:
                plt.imshow(person)
            else:
                plt.imshow(cv2.cvtColor(person, cv2.COLOR_BGR2RGB))

            # we compute and plot our histograms on the same chart
            plt.subplot(2, 2, k+1)
            for channel, color in enumerate(colors):
                hist = cv2.calcHist([person], [channel], None, [histSize], histRange)
                plt.plot(hist, color)
                

            plt.axis('on')
            plt.xlabel('Number of bins')
            plt.ylabel('Number of pixels')
    k+=2



#we save our results
if IF_HSV:
    plt.savefig(os.path.join(images_path, 'Skate histograms HSV.jpg'))
else:
    plt.savefig(os.path.join(images_path, 'Skate histograms RGB.jpg'))

plt.show()
