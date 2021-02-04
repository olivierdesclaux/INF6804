import numpy as np
import json
import os
import cv2
import matplotlib.pyplot as plt
gt_path = 'data/part1/gt.json'
images_path = 'data/part1/images'
save_path = 'data/part1/results'

if not os.path.isdir(save_path):
    os.mkdir(save_path)

# We extract the data from our json
with open(gt_path) as f:
    gt = json.load(f)
annotations = gt['annotations']
categories = gt['categories']

# we extract the ski image
ski_img = cv2.imread(os.path.join(images_path, 'ski.jpg'), 1)

annotations = [ann for ann in annotations if ann['image'] == 'ski'] #we get only the ski annotations

# En regardant dans les catégories, on voit que l'Id d'une personne est 1 (supercategory 'person'). 


k = 1 # for graphic purposes only

#Histogram settings
histSize = 256 # number of bins
histRange = [0, 256]
colors = ['b', 'g', 'r']

plt.figure(figsize = (15, 10))


for ann in annotations:
    if ann['category_id'] == 1: #category_id of a person
        
        #extract roi
        x, y, w, h = np.array(ann['bbox']).astype('int32')
        person = ski_img[y: y+h, x: x+w]
        
        # we show the person
        plt.subplot(2, 2, k)
        plt.axis('off')
        plt.imshow(cv2.cvtColor(person, cv2.COLOR_BGR2RGB))

        # we compute and plot our histograms on the same chart
        plt.subplot(2, 2, k+1)
        plt.axis('on')
        for channel, color in enumerate(colors): #range(3):
            # hist = cv2.calcHist([person], [channel], None, [histSize], histRange)
            hist, _ = np.histogram(person[:,:,channel], 255)
            plt.plot(hist, color)#colors[channel])
        plt.xlabel('Number of bins')
        plt.ylabel('Number of pixels')
        plt.legend(('Blue', 'Green', 'Red'))
        k+=2



#we save our results0
plt.savefig(os.path.join(save_path, 'Ski histograms.jpg'))

plt.show()
