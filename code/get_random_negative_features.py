import numpy as np
import os
import random
import math
from cyvlfeat.hog import hog
from skimage.io import imread
from skimage.transform import pyramid_gaussian
from skimage import color
from tqdm import tqdm

# you may implement your own data augmentation functions

def get_random_negative_features(non_face_scn_path, feature_params, num_samples):
    '''
    FUNC: This funciton should return negative training examples (non-faces) from
        any images in 'non_face_scn_path'. Images should be converted to grayscale,
        because the positive training data is only available in grayscale. For best
        performance, you should sample random negative examples at multiple scales.
    ARG:
        - non_face_scn_path: a string; directory contains many images which have no
                             faces in them.
        - feature_params: a dict; with keys,
                          > template_size: int (probably 36); the number of
                            pixels spanned by each train/test template.
                          > hog_cell_size: int (default 6); the number of pixels
                            in each HoG cell. 
                          Template size should be evenly divisible by hog_cell_size.
                          Smaller HoG cell sizez tend to work better, but they 
                          make things slower because the feature dimenionality 
                          increases and more importantly the step size of the 
                          classifier decreases at test time.
    RET:
        - features_neg: (N,D) ndarray; N is the number of non-faces and D is 
                        the template dimensionality, which would be, 
                        (template_size/hog_cell_size)^2 * 31,
                        if you're using default HoG parameters.
        - neg_examples: TODO
    '''
    #########################################
    ##          you code here              ##
    #########################################
    template_size = feature_params['template_size']
    hog_cell_size = feature_params['hog_cell_size']
    features_neg = []
    images_path = []
    print('Load negative features')
    for image_path in os.listdir(non_face_scn_path):
        if image_path.find('.jpg') != -1:
            image_path = os.path.join(non_face_scn_path, image_path)
            images_path.append(image_path)

    for image_path in tqdm(images_path):
        im = imread(image_path, as_grey = True)
        im_pyramids = tuple(pyramid_gaussian(im))
        for im_pyramid in im_pyramids[:1]:
            num_sample_per_image =  num_samples // len(images_path) + 1
            if min(im_pyramid.shape[0], im_pyramid.shape[1]) <= template_size:
                break
            elif min(im_pyramid.shape[0], im_pyramid.shape[1]) < template_size + num_sample_per_image:
                num_sample_per_image =  min(im_pyramid.shape[0], im_pyramid.shape[1]) - template_size
            height_list = np.random.choice(im_pyramid.shape[0]-template_size, int(num_sample_per_image), replace = False)
            weight_list = np.random.choice(im_pyramid.shape[1]-template_size, int(num_sample_per_image), replace = False)
            for height, weight in zip(height_list, weight_list):
                features_neg.append(np.reshape(hog(im_pyramid[height : height + template_size, weight:weight + template_size], hog_cell_size), -1))

    features_neg = np.array(features_neg)
    print('Number of training examples: {0}' .format(len(features_neg)))
    neg_examples = len(features_neg)
    #########################################
    ##          you code here              ##
    #########################################
            
    return features_neg, neg_examples

