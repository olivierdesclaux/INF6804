import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import os
import sys



def main(args):
    
    image_path = os.path.join(args.data_path, args.image)
    image = cv2.imread(image_path) # caution, image is read as bgr ! 
    method = args.method
    database = os.path.join(args.data_path, 'database')

    top_n = 5 # number of best matching images
    best_results = compare_histogram(image, database, top_n)
    fig1, axs = plt.subplots(ncols=top_n +1, nrows=1, constrained_layout=True, figsize=(20, 8))

    axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0].axis('off')
    axs[0].set_title('Query image')
    for i, res in enumerate(best_results):
        axs[i+1].imshow(cv2.cvtColor(cv2.imread(os.path.join(database, res)), cv2.COLOR_BGR2RGB))
        axs[i+1].set_title(res)
        axs[i+1].axis('off')

    plt.show()
    # print(histogram(image))
    # print(cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:,:,0])
    # print(compare_histogram(image, database))

def histogram(query):
    """
    Input:
    query: np.array, bgr query image 
    database: str, relative path to the database where all the other images are stored

    Output:

    """
    target_width = 256
    target_height = 256
    query = cv2.resize(query, (target_width, target_height))
    query_hsv = cv2.cvtColor(query, cv2.COLOR_BGR2HSV)
    # print(type(query_hsv[0,0,0]))
    # print(np.max(query_hsv[:, :, 2]))
    h_bins = np.int16(np.dot([1, 26, 41, 121, 191, 271, 295, 316, 361], 0.5))
    s_bins = np.int16(np.dot([0, 0.2, 0.7, 1], 255))
    v_bins = np.int16(np.dot([0, 0.2, 0.7, 1], 255))

    h_hist, _ = np.histogram(query_hsv[:,:,0].ravel(), bins = h_bins)
    s_hist, _ =  np.histogram(query_hsv[:,:,1], bins = s_bins)
    v_hist, _ = np.histogram(query_hsv[:,:,2], bins = v_bins)
    print( np.array(h_hist.tolist() +  s_hist.tolist() + v_hist.tolist()))
    return np.array(h_hist.tolist() +  s_hist.tolist() + v_hist.tolist())

    # return h_hist + s_hist + v_hist
    # return np.concatenate((h_hist, s_hist, v_hist))


def compare_histogram(query, database, top_n):
    """
    This implementation is based on paper "Content-based image retrieval using color and texture fused features"
    https://www.sciencedirect.com/science/article/pii/S0895717710005352
    
    Input:
        query: np.array, bgr query image 
        database: str, relative path to the database where all the other images are stored

    """

    query_hist = histogram(query)
    data = os.listdir(database)
    distances = np.zeros(len(data)) #initialise distances
    for i, x in enumerate(data):
        print(x)
        x = os.path.join(database, x)
        target_hist = histogram(cv2.imread(x))
        distances[i] = np.linalg.norm(query_hist - target_hist)
    
    print(distances)
    res = [data[i]for i in np.argsort(distances)[:top_n]]
    return res
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type = str, help='Path to image')
    parser.add_argument('--method', type = str,  default='hist', help='Hist or Orb method')
    parser.add_argument('--data_path', type=str, default='./data/part2')

    args = parser.parse_args()

    main(args)
