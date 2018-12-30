from sklearn import preprocessing
from sklearn.svm import LinearSVC, SVC
from sklearn import preprocessing
import pdb
from sklearn.model_selection import cross_val_score


def svm_classify(train_image_feats, train_labels, test_image_feats):
    #################################################################################
    # TODO :                                                                        #
    # This function will train a set of linear SVMs for multi-class classification  #
    # and then use the learned linear classifiers to predict the category of        #
    # every test image.                                                             # 
    #################################################################################
    ##################################################################################
    # NOTE: Some useful functions                                                    #
    # LinearSVC :                                                                    #
    #   The multi-class svm classifier                                               #
    #        e.g. LinearSVC(C= ? , class_weight=None, dual=True, fit_intercept=True, #
    #                intercept_scaling=1, loss='squared_hinge', max_iter= ?,         #
    #                multi_class= ?, penalty='l2', random_state=0, tol= ?,           #
    #                verbose=0)                                                      #
    #                                                                                #
    #             C is the penalty term of svm classifier, your performance is highly#
    #          sensitive to the value of C.                                          #
    #   Train the classifier                                                         #
    #        e.g. classifier.fit(? , ?)                                              #
    #   Predict the results                                                          #
    #        e.g. classifier.predict( ? )                                            #
    ##################################################################################
    '''
    Input : 
        train_image_feats : training images features
        train_labels : training images labels
        test_image_feats : testing images features
    Output :
        Predict labels : a list of predict labels of testing images (Dtype = String).
    '''
    
    clf = LinearSVC()
    #C=0.005,random_state=0, tol=1e-5
    #This for RBF kernel
    #clf = SVC(kernel = 'rbf', random_state=0, gamma = 0.2, C=10)
    clf.fit(train_image_feats, train_labels)
    pred_label = clf.predict(test_image_feats)
    
    sc = cross_val_score(clf, train_image_feats, train_labels)
    print 'Validation Score:',sc.mean()
    #############################################################################
    #                                END OF YOUR CODE                           #
    #############################################################################
    return pred_label
