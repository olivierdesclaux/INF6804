import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import os
import glob
import sys



def main(args):

    ############### Reading Input #########################

    if args.all_images == True:
        image_path_list = glob.glob(args.data_path + '*.jpg')
    else:
        image_path_list = [os.path.join(args.data_path, args.image)]


    method = eval(args.method)
    n_images = len(image_path_list)
    database = os.path.join(args.data_path, 'database')
    top_n = args.top_n


    ############### Setting pyplot environment ##############
    fig1, axs = plt.subplots(ncols=top_n +1, nrows=n_images, constrained_layout=True, figsize=(20, 8))
    if n_images == 1:
        axs = np.reshape(axs, (1, np.size(axs)))

    ############### Processing images ######################  
    for idx, image_path in enumerate(image_path_list):
        image = cv2.imread(image_path) # caution, image is read as bgr ! 
        image_preprocessed = preprocess(image)
        best_results = compare_histogram(image_preprocessed, database, top_n, method)

        axs[idx, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # axs[idx, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
        axs[idx, 0].axis('off')
        axs[idx, 0].set_title('Query image')


        for i, res in enumerate(best_results):
            axs[idx, i+1].imshow(cv2.cvtColor(cv2.imread(os.path.join(database, res)), cv2.COLOR_BGR2RGB))
            axs[idx, i+1].set_title(res)
            axs[idx, i+1].axis('off')

    plt.show()

def histogram(query):
    """
    Input:
        query: np.array, query image 
    Output:
        res: np.array, one-dimensional vector with all the histograms merged into one

    """
    h_bins = np.int16(np.dot([1, 26, 41, 121, 191, 271, 295, 316, 361], 0.5))
    s_bins = np.int16(np.dot([0, 0.2, 0.7, 1], 255))
    v_bins = np.int16(np.dot([0, 0.2, 0.7, 1], 255))

    h_hist, _ = np.histogram(query[:,:,0].ravel(), bins = h_bins)
    s_hist, _ =  np.histogram(query[:,:,1].ravel(), bins = s_bins)
    v_hist, _ = np.histogram(query[:,:,2].ravel(), bins = v_bins)


    h_hist = np.dot(h_hist, 1/np.sum(h_hist))
    s_hist = np.dot(s_hist, 1/np.sum(s_hist))
    v_hist = np.dot(v_hist, 1/np.sum(v_hist))


    res =  np.array(h_hist.tolist() +  s_hist.tolist() + v_hist.tolist())


    return res


def preprocess(image, RESIZE = False):
    """
    Function for preprocessing our image

    input: 
        np.array, image 

    output:
        res: np.array, preprocessed image
    """
    
    # Change color code from BGR to HSV
    res = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Preprocessing steps to be added
    
    if RESIZE:
        target_width = 256
        target_height = 256
        res = cv2.resize(res, (target_width, target_height))

    return res

def block_histogram(query):
    """
    This function divides the query image into blocks and computes histograms on these blocks. 
    This implementation is based on paper "Content-based image retrieval using color and texture fused features"
    https://www.sciencedirect.com/science/article/pii/S0895717710005352

    Input:
        query: np.array, image

    output:
        res: np.array. If we split our image in N blocks, dimension will be Number of Hist bins * N
    """
    h,w, _ = np.shape(query)
    
    n_blocks = (3,3)
    height_indices = [k * int(h/n_blocks[0]) for k in range(n_blocks[0] + 1)]
    width_indices = [k * int(w/n_blocks[1]) for k in range(n_blocks[1] + 1)]

    # block_weights = np.array([1, 2, 1, 2, 4, 2, 1, 2, 1])
    # block_weights = block_weights/np.sum(block_weights)

    block_weights = np.ones(9)
    res = np.zeros((n_blocks[0]*n_blocks[1], 14))
    cnt = 0

    for i in range(n_blocks[0]):
        for j in range (n_blocks[1]):
            # divide image into blocks
            block = query[height_indices[i]: height_indices[i+1], width_indices[j]: width_indices[j+1]]
            hist = histogram(block)
            res[cnt] = hist
            cnt += 1

    res = np.matmul(block_weights, res)
    return res


def weighted_block_histogram(query):
    h,w, _ = np.shape(query)
    height_indices = [0, int(h/4), int(3*h/4), h]
    width_indices = [0, int(w/4), int(3*w/4), w]
    block_weights = np.array([1, 2, 1, 2, 4, 2, 1, 2, 1])

    for i in range(3):
        for j in range (3):
            # divide image into blocks
            block = query[height_indices[i]: height_indices[i+1], width_indices[j]: width_indices[j+1]]
            hist = histogram(block)
            res[cnt] = hist
            cnt += 1

    

def compare_histogram(query, database, top_n, method):
    """
    This implementation is based on paper "Content-based image retrieval using color and texture fused features"
    https://www.sciencedirect.com/science/article/pii/S0895717710005352
    
    Input:
        query: np.array, bgr query image 
        database: str, relative path to the database where all the other images are stored

    """ 
    # if method.__name__ == 'block_histogram':
    #     query = preprocess(query, RESIZE = False)
    # elif method.__name__ == 'histogram':
    #     target_im = preprocess(query, RESIZE = False)
    
    query_hist = method(query)
    data = os.listdir(database)
    distances = np.zeros(len(data)) #initialise distances

    for i, x in enumerate(data):
        x = os.path.join(database, x)
        target_im = cv2.imread(x)
        if method.__name__ == 'block_histogram':
            target_im = preprocess(target_im, RESIZE = False)
        elif method.__name__ == 'histogram':
            target_im = preprocess(target_im, RESIZE = False)
        target_hist = method(target_im)

        # print(target_hist)
        distances[i] = np.linalg.norm(query_hist - target_hist)
    print(distances)
    res = [data[i]for i in np.argsort(distances)[:top_n]]
    return res
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type = str, help='Image name', default = 'airplane_query.jpg')
    parser.add_argument('--all_images', type = bool, default = False, help= 'Set to true if process all images')
    parser.add_argument('--method', default='histogram', help='histogram, block_histogram or Orb method')
    parser.add_argument('--data_path', type=str, default='./data/part2/')
    parser.add_argument('--top_n', type = int, default = 5, help = 'Top n results we wan to show')

    args = parser.parse_args()

    main(args)
