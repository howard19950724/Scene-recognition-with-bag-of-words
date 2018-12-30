import numpy as np
import os
from skimage.io import imread
from skimage.transform import resize
from skimage import color
from cyvlfeat.hog import hog

from non_max_supr_bbox import non_max_supr_bbox

def run_detector(test_scn_path, model, feature_params):
    """
    FUNC: This function returns detections on all of the images in a given path.
        You will want to use non-maximum suppression on your detections or your
        performance will be poor (the evaluation counts a duplicate detection as
        wrong). The non-maximum suppression is done on a per-image basis. The
        starter code includes a call to a provided non-max suppression function.
    ARG:
        - test_scn_path: a string; This directory contains images which may or
                        may not have faces in them. This function should work for
                        the MIT+CMU test set but also for any other images.
        - model: the linear classifier model
        - feature_params: a dict; 'template_size': the number of pixels spanned
                        by each train / test template (probably 36).
                        'hog_cell_size': the number of pixels in each HoG cell
                        (default 6).
                        Template size should be evenly divisible by hog_cell_size.
                        Smaller HoG cell sizes tend to work better, but they make
                        things slower because the feature dimensionality increases
                        and more importantly the step size of the classifier
                        decreases at test time.
        - MORE...  You can add additional arguments for advanced work like multi-
                   scale, pixel shift and so on.
                   
    RET:
        - bboxes: (N, 4) ndarray; N is the number of non-overlapping detections, bboxes[i,:] is
                        [x_min, y_min, x_max, y_max] for detection i.
        - confidences: (N, 1) ndarray; confidences[i, :] is the real valued confidence
                        of detection i.
        - image_ids: (N, 1) ndarray;  image_ids[i, :] is the image file name for detection i.
    """
    # The placeholder version of this code will return random bounding boxes in
    # each test image. It will even do non-maximum suppression on the random
    # bounding boxes to give you an example of how to call the function.

    # Your actual code should convert each test image to HoG feature space with
    # a _single_ call to vl_hog for each scale. Then step over the HoG cells,
    # taking groups of cells that are the same size as your learned template,
    # and classifying them. If the classification is above some confidence,
    # keep the detection and then pass all the detections for an image to
    # non-maximum suppression. For your initial debugging, you can operate only
    # at a single scale and you can skip calling non-maximum suppression.

    test_images = os.listdir(test_scn_path)

    # initialize these as empty and incrementally expand them.
    bboxes = np.zeros([0, 4])
    confidences = np.zeros([0, 1])
    image_ids = np.zeros([0, 1])

    cell_size = feature_params['hog_cell_size']
    cell_num = feature_params['template_size'] / feature_params['hog_cell_size']  # cell number of each template

    for i in range(len(test_images)):

        #########################################
        ##          you code here              ##
        #########################################
        cur_bboxes = np.zeros([0, 4])
        cur_confidences = np.zeros([0, 1])
        cur_image_ids = np.zeros([0, 1])
        
        image_path = os.path.join(test_scn_path, test_images[i])
        img = imread(image_path, as_grey=True)
        scale_arr = np.arange(1.0, 0, -0.1)
        for scale in scale_arr:
            img_resize = resize(img, [int(img.shape[0]*scale), int(img.shape[1] * scale)])
            test_image_hog = hog(img_resize, cell_size)
            for h in range(test_image_hog.shape[0] - cell_num + 1):
                for w in range(test_image_hog.shape[1] - cell_num + 1):
                    hog_window = np.reshape(test_image_hog[h:h + cell_num, w:w + cell_num, :], (1, -1))
                    score = model.decision_function(hog_window)
                    if score > -0.5:
                        min_y = int(h * cell_size / scale)
                        min_x = int(w * cell_size / scale)
                        max_y = int((h + cell_num) * cell_size / scale)
                        max_x = int((w + cell_num) * cell_size / scale)
                        
                        cur_confidence = np.array([score])
                        cur_image_id = np.array([[test_images[i]]])
                        cur_bbox = np.array([[min_x, min_y, max_x, max_y]])
                        
                        cur_bboxes = np.concatenate([cur_bboxes, cur_bbox], 0)
                        cur_confidences = np.concatenate([cur_confidences, cur_confidence], 0)
                        cur_image_ids = np.concatenate([cur_image_ids, cur_image_id], 0)
    
        #########################################
        ##          you code here              ##
        #########################################

        # non_max_supr_bbox can actually get somewhat slow with thousands of
        # initial detections. You could pre-filter the detections by confidence,
        # e.g. a detection with confidence -1.1 will probably never be
        # meaningful. You probably _don't_ want to threshold at 0.0, though. You
        # can get higher recall with a lower threshold. You don't need to modify
        # anything in non_max_supr_bbox, but you can.
        is_maximum = non_max_supr_bbox(cur_bboxes, cur_confidences, img.shape)

        cur_bboxes = cur_bboxes[is_maximum[:, 0], :]
        cur_confidences = cur_confidences[is_maximum[:, 0], :]
        cur_image_ids = cur_image_ids[is_maximum[:, 0]]

        bboxes = np.concatenate([bboxes, cur_bboxes], 0)
        confidences = np.concatenate([confidences, cur_confidences], 0)
        image_ids = np.concatenate([image_ids, cur_image_ids], 0)

    return bboxes, confidences, image_ids

