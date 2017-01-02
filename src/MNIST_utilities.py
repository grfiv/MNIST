#!/usr/bin/python
# -*- coding: utf-8 -*-

def load_all_MNIST(portion=1.0, test_deskewed=True, create_X_copy=False):
    """
    =========================================================
    Load the MNIST training and test data
    =========================================================

    Args:
        portion (float): a number in the range (0.0, 1.0]
                         the proportion of the data to be returned
                         
        test_deskewed (bool): if True testX contains the deskewed images
                              else    testX contains the original images
                              
        create_X_copy (bool): if True add trainXoriginal, textXoriginal
                              to the return tuple
                              The purpose is to preserve the original data
                              in the event you scale, do PCA, etc. but later
                              want to look at the images

    Returns:
        tuple: (trainX, trainY, testX, testY)
               Note: the data is shuffled before it is returned
               
    Raises:
        ValueError: if portion is not in the (0.0, 1.0] range
    """
    import csv
    from sklearn.utils import shuffle
    import numpy as np
    
    # test portion argument
    # =====================
    if portion <= 0.0 or portion > 1.0:
        raise ValueError('load_all_MNIST argument portion is not in (0.0, 1.0] range')
    
    # where's the data?
    # =================
    file_path = '/home/george/Dropbox/MNIST/data/'

    train_img_deskewed_filename = 'train-images_deskewed.csv'
    train_img_original_filename = 'train-images.csv'

    if test_deskewed:
        test_img_filename  = 't10k-images_deskewed.csv'
    else:
        test_img_filename  = 't10k-images.csv'
        
    train_label_filename   = 'train-labels.csv'
    test_label_filename    = 't10k-labels.csv'
    
    # Read the training images and labels, both original and deskewed
    # ===============================================================
    # read both trainX files
    with open(file_path + train_img_original_filename,'r') as f:
        data_iter = csv.reader(f, delimiter = ',')
        data      = [data for data in data_iter]
    trainXo = np.ascontiguousarray(data, dtype = np.float64)  

    with open(file_path + train_img_deskewed_filename,'r') as f:
        data_iter = csv.reader(f, delimiter = ',')
        data      = [data for data in data_iter]
    trainXd = np.ascontiguousarray(data, dtype = np.float64)

    # vertically concatenate the two files
    trainX = np.vstack((trainXo, trainXd))

    trainXo = None
    trainXd = None

    # read trainY twice and vertically concatenate
    with open(file_path + train_label_filename,'r') as f:
        data_iter = csv.reader(f, delimiter = ',')
        data      = [data for data in data_iter]
    trainYo = np.ascontiguousarray(data, dtype = np.int8) 
    trainYd = np.ascontiguousarray(data, dtype = np.int8)

    trainY = np.vstack((trainYo, trainYd)).ravel()

    trainYo = None
    trainYd = None
    data    = None

    # shuffle trainX & trainY
    trainX, trainY = shuffle(trainX, trainY, random_state=0)

    # use less data if specified
    if portion < 1.0:
        trainX = trainX[:portion*trainX.shape[0]]
        trainY = trainY[:portion*trainY.shape[0]]
        
    # Read the test images and labels
    # ===============================
    # read testX
    with open(file_path + test_img_filename,'r') as f:
        data_iter = csv.reader(f, delimiter = ',')
        data      = [data for data in data_iter]
    testX = np.ascontiguousarray(data, dtype = np.float64)  

    # read testY
    with open(file_path + test_label_filename,'r') as f:
        data_iter = csv.reader(f, delimiter = ',')
        data      = [data for data in data_iter]
    testY = np.ascontiguousarray(data, dtype = np.int8)
    
    data = None

    # shuffle testX, testY
    testX, testY = shuffle(testX, testY, random_state=0)

    # use a smaller dataset if specified
    if portion < 1.0:
        testX = testX[:portion*testX.shape[0]]
        testY = testY[:portion*testY.shape[0]]
        
    # return the data
    # ===============
    print("trainX shape: {0}".format(trainX.shape))
    print("trainY shape: {0}".format(trainY.shape))

    print("\ntestX shape: {0}".format(testX.shape))
    print("testY shape: {0}".format(testY.shape))
    
    if create_X_copy:
        trainXoriginal = trainX.copy()
        testXoriginal  = testX.copy()
        return trainX, trainY, testX, testY, trainXoriginal, testXoriginal
    else:
        return trainX, trainY, testX, testY
    
# ====================================================================================

def load_Kaggle(portion=1.0, test_deskewed=True, create_X_copy=False):
    """
    =========================================================
    Load the Kaggle digit contest training and test data
    =========================================================

    Args:
        portion (float): a number in the range (0.0, 1.0]
                         the proportion of the data to be returned
                         
        test_deskewed (bool): if True testX contains the deskewed images
                              else    testX contains the original images
                              
        create_X_copy (bool): if True add trainXoriginal, textXoriginal
                              to the return tuple
                              The purpose is to preserve the original data
                              in the event you scale, do PCA, etc. but later
                              want to look at the images

    Returns:
        tuple: (trainX, trainY, testX) or
               (trainX, trainY, testX, trainXoriginal, textXoriginal)
               Note: the data is shuffled before it is returned
               Note: testY is not returned
               
    Raises:
        ValueError: if portion is not in the (0.0, 1.0] range
    """
    import csv
    from sklearn.utils import shuffle
    import numpy as np
    
    # test portion argument
    # =====================
    if portion <= 0.0 or portion > 1.0:
        raise ValueError('load_Kaggle argument portion is not in (0.0, 1.0] range')
    
    # where's the data?
    # =================
    file_path = '../kaggle/data/'

    train_img_deskewed_filename = 'kaggle_trainX_deskewed.csv'
    train_img_original_filename = 'kaggle_trainX_original.csv'
    train_label_filename        = 'kaggle_trainY.csv'
     
    if test_deskewed:
        test_img_filename  = 'kaggle_testX_deskewed.csv'
    else:
        test_img_filename  = 'kaggle_testX.csv'
    
    
    # Read the training images and labels, both original and deskewed
    # ===============================================================
    # read both trainX files
    with open(file_path + train_img_original_filename,'r') as f:
        data_iter = csv.reader(f, delimiter = ',')
        data      = [data for data in data_iter]
    trainXo = np.ascontiguousarray(data, dtype = np.float64)  

    with open(file_path + train_img_deskewed_filename,'r') as f:
        data_iter = csv.reader(f, delimiter = ',')
        data      = [data for data in data_iter]
    trainXd = np.ascontiguousarray(data, dtype = np.float64)

    # vertically concatenate the two files
    trainX = np.vstack((trainXo, trainXd))

    trainXo = None
    trainXd = None

    # read trainY, previously vertically stacked
    with open(file_path + train_label_filename,'r') as f:
        data_iter = csv.reader(f, delimiter = ',')
        data      = [data for data in data_iter]
    trainY = np.ascontiguousarray(data, dtype = np.int8).ravel()
    
    data    = None

    # shuffle trainX & trainY
    trainX, trainY = shuffle(trainX, trainY, random_state=0)
        
    # Read the test images and labels
    # ===============================
    # read testX
    with open(file_path + test_img_filename,'r') as f:
        data_iter = csv.reader(f, delimiter = ',')
        data      = [data for data in data_iter]
    testX = np.ascontiguousarray(data, dtype = np.float64)  
    
    data = None

    # use less data if specified
    # ==========================
    if portion < 1.0:
        trainX = trainX[:portion*trainX.shape[0]]
        trainY = trainY[:portion*trainY.shape[0]]
        testX  = testX[:portion*testX.shape[0]]
        
    # return the data
    # ===============
    print("trainX shape: {0}".format(trainX.shape))
    print("trainY shape: {0}".format(trainY.shape))

    print("\ntestX shape: {0}".format(testX.shape))
    
    if create_X_copy:
        trainXoriginal = trainX.copy()
        testXoriginal  = testX.copy()
        return trainX, trainY, testX, trainXoriginal, testXoriginal
    else:
        return trainX, trainY, testX


# ====================================================================================


def print_imgs(images, actual_labels, predicted_labels, starting_index = 0, size=6):
    """
    print a grid of images
    showing any differences in predicted values
    
    images           m x n array of pixels, n assumed to be a perfect square
    actual_labels    m x 1 array of the actual labels
    predicted_labels m x 1 of predicted labels
    starting_index   scalar, where in 1...m to start
    size             scalar the grid of images is size x size
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    img_dim  = int(pow(images.shape[1],0.5)) # images assumed to be square
    
    fig, axs = plt.subplots(size,size, figsize=(img_dim,img_dim), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = 0.0001, wspace=.001)
    axs = axs.ravel()
    
    # create a view with the 1 x flat bit strings reshaped to img_dim x img_dim matrices
    imgs = images.view().reshape(images.shape[0], img_dim, img_dim)
    
    for grid_i, img_i in enumerate(xrange(starting_index, starting_index+(size*size))):
        
        axs[grid_i].imshow(imgs[img_i], cmap=plt.cm.gray_r, interpolation='nearest')
        
        if actual_labels[img_i] != predicted_labels[img_i]:
            axs[grid_i].set_title("actual: {0}; predicted: {1}" \
                                  .format(actual_labels[img_i], predicted_labels[img_i]), 
                                  fontsize=16,
                                  color='r')
        else:
            axs[grid_i].set_title("label: {0}" \
                                  .format(actual_labels[img_i]), 
                                  fontsize=16)
            
    plt.show()


# ====================================================================================

def load_module(code_path):
    """
    =========================
    Dynamically load a module
    =========================
    
    see: http://code.davidjanes.com/blog/2008/11/27/how-to-dynamically-load-python-code/

    Args:
        code_path (str): the path to a .py file that you want to load

    Returns:
        module: a call to 
                   m = load_module('path/to/foo.py')
                allows calls to 
                   m.def_name(params)
                   
                where def_name is a function defined in foo.py
               
    Raises:
        ImportError: if the file isn't there
    """
    import md5
    import os.path
    import imp
    import traceback
    try:
        try:
            code_dir = os.path.dirname(code_path)
            code_file = os.path.basename(code_path)

            fin = open(code_path, 'rb')

            return  imp.load_source(md5.new(code_path).hexdigest(), code_path, fin)
        finally:
            try: fin.close()
            except: pass
    except ImportError, x:
        traceback.print_exc(file = sys.stderr)
        raise
    except:
        traceback.print_exc(file = sys.stderr)
        raise
