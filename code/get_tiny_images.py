from PIL import Image

import pdb
import numpy as np

def get_tiny_images(image_paths):

    '''
    Input : 
        image_paths: a list(N) of string where where each string is an image 
        path on the filesystem.
    Output :
        tiny image features : (N, d) matrix of resized and then vectorized tiny
        images. E.g. if the images are resized to 16x16, d would equal 256.
    '''
    #############################################################################
    # TODO:                                                                     #
    # To build a tiny image feature, simply resize the original image to a very #
    # small square resolution, e.g. 16x16. You can either resize the images to  #
    # square while ignoring their aspect ratio or you can crop the center       #
    # square portion out of each image. Making the tiny images zero mean and    #
    # unit length (normalizing them) will increase performance modestly.        #
    #############################################################################
    
    width = height = 16
    
    tiny_images = np.zeros((len(image_paths), width * height), dtype = 'float32')
    
    for num in range(len(image_paths)):
        img = Image.open(image_paths[num])
        img_resize = np.asarray(img.resize((width, height), Image.ANTIALIAS), dtype='float32')
        img_resize = img_resize.flatten()
        img_tiny = (img_resize-np.mean(img_resize))/np.std(img_resize)
        tiny_images[num, :] = img_tiny
    
    ##############################################################################
    #                                END OF YOUR CODE                            #
    ##############################################################################

    return tiny_images
