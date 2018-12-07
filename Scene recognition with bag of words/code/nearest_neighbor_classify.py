from __future__ import print_function

import numpy as np
import scipy.spatial.distance as distance

def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats):
    ###########################################################################
    # TODO:                                                                   #
    # This function will predict the category for every test image by finding #
    # the training image with most similar features. Instead of 1 nearest     #
    # neighbor, you can vote based on k nearest neighbors which will increase #
    # performance (although you need to pick a reasonable value for k).       #
    ###########################################################################
    ###########################################################################
    # NOTE: Some useful functions                                             #
    # distance.cdist :                                                        #
    #   This function will calculate the distance between two list of features#
    #       e.g. distance.cdist(? ?)                                          #
    ###########################################################################
    '''
    Input : 
        train_image_feats : 
            image_feats is an (N, d) matrix, where d is the 
            dimensionality of the feature representation.

        train_labels : 
            image_feats is a list of string, each string
            indicate the ground truth category for each training image. 

        test_image_feats : 
            image_feats is an (M, d) matrix, where d is the 
            dimensionality of the feature representation.
    Output :
        test_predicts : 
            a list(M) of string, each string indicate the predict
            category for each testing image.
    '''
    test_predicts = []
    for num in range(test_image_feats.shape[0]):
        each_row = []
        each = np.tile(test_image_feats[num],(train_image_feats.shape[0],1))
        square_each = np.square(each - train_image_feats)
        for sq in range(square_each.shape[0]):
            each_row.append(np.sqrt(sum(square_each[sq])))
        minimum = min(each_row)
        minimum_ind = each_row.index(min(each_row))
        test_predicts.append(train_labels[minimum_ind])
    #############################################################################
    #                                END OF YOUR CODE                           #
    #############################################################################
    return test_predicts
