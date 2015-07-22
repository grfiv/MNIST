# ======================================
#
# script run on AWS c4.4xlarge
#
# ======================================
from __future__ import division
import os, time, math, csv
import cPickle as pickle

import matplotlib.pyplot as plt

import numpy as np

from sklearn.preprocessing    import StandardScaler
from sklearn.utils            import shuffle

from sklearn.svm              import SVC

from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split
from sklearn.grid_search      import RandomizedSearchCV

from sklearn.metrics          import classification_report, confusion_matrix

np.random.seed(seed=1009)


file_path = '../data/'

DESKEWED = True
if DESKEWED:
    train_img_filename = 'train-images_deskewed.csv'
    test_img_filename  = 't10k-images_deskewed.csv'
else:
    train_img_filename = 'train-images.csv'
    test_img_filename  = 't10k-images.csv'
    
train_label_filename   = 'train-labels.csv'
test_label_filename    = 't10k-labels.csv'


# ##Read the training images and labels

# In[113]:

# read trainX
with open(file_path + train_img_filename,'r') as f:
    data_iter = csv.reader(f, delimiter = ',')
    data      = [data for data in data_iter]
trainX = np.ascontiguousarray(data, dtype = np.float64)  

# scale trainX
scaler = StandardScaler()
scaler.fit(trainX)                 # find mean/std for trainX
trainX = scaler.transform(trainX)  # scale trainX with trainX mean/std

# read trainY
with open(file_path + train_label_filename,'r') as f:
    data_iter = csv.reader(f, delimiter = ',')
    data      = [data for data in data_iter]
trainY = np.ascontiguousarray(data, dtype = np.int8).ravel() 

    
# shuffle trainX & trainY
trainX, trainY = shuffle(trainX, trainY, random_state=0)


# ##Read the test images and labels

# In[114]:

# read testX
with open(file_path + test_img_filename,'r') as f:
    data_iter = csv.reader(f, delimiter = ',')
    data      = [data for data in data_iter]
testX = np.ascontiguousarray(data, dtype = np.float64)  

# scale testX
testX = scaler.transform(testX)    # scale testX with trainX mean/std


# read testY
with open(file_path + test_label_filename,'r') as f:
    data_iter = csv.reader(f, delimiter = ',')
    data      = [data for data in data_iter]
testY = np.ascontiguousarray(data, dtype = np.int8).ravel()


# shuffle testX, testY
testX, testY = shuffle(testX, testY, random_state=0)



# #SVC Default Parameter Settings

# In[ ]:

# default parameters for SVC
# ==========================
default_svc_params = {}

default_svc_params['C']            = 1.0      # penalty
default_svc_params['class_weight'] = None     # Set the parameter C of class i to class_weight[i]*C
                                              # set to 'auto' for unbalanced classes
default_svc_params['gamma']        = 0.0      # Kernel coefficient for 'rbf', 'poly' and 'sigmoid'

default_svc_params['kernel']       = 'rbf'    # 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or a callable 
default_svc_params['shrinking']    = True     # Whether to use the shrinking heuristic.     
default_svc_params['probability']  = False    # Whether to enable probability estimates.    
default_svc_params['tol']          = 0.001    # Tolerance for stopping criterion. 
default_svc_params['cache_size']   = 200      # size of the kernel cache (in MB).

default_svc_params['max_iter']     = -1       # limit on iterations within solver, or -1 for no limit. 

#default_svc_params['random_state'] = 1009    
default_svc_params['verbose']      = False 
default_svc_params['degree']       = 3        # 'poly' only
default_svc_params['coef0']        = 0.0      # 'poly' and 'sigmoid' only

# set parameters for the classifier
# =================================
svc_params = dict(default_svc_params)

svc_params['cache_size']  = 2000
#svc_params['probability'] = True

svc_params['kernel']     = 'poly'
svc_params['C']          = 1.0
svc_params['gamma']      = 0.0
svc_params['degree']     = 3
svc_params['coef0']      = 1

# the classifier
# ==============
svc_clf = SVC(**svc_params)


# ##RANDOMIZED grid search

# In[ ]:

t0 = time.time()

# search grid
# ===========
search_grid = dict(C      = np.logspace( 0,  5, 50),
                   gamma  = np.logspace(-5, -1, 50),
                   degree = [2, 3, 4, 5, 6, 7, 8, 9])

# for coef0, see http://stackoverflow.com/questions/21390570/scikit-learn-svc-coef0-parameter-range
# but also see   http://www.eric-kim.net/eric-kim-net/posts/1/kernel_trick.html

# stratified K-Fold indices
# =========================
SKFolds = StratifiedKFold(y            = trainY, 
                          n_folds      = 3, 
                          indices      = None) 

# default parameters for RandomizedSearchCV
# =========================================
default_random_params = {}
default_random_params['scoring']      = None            
default_random_params['fit_params']   = None       # dict of parameters to pass to the fit method
default_random_params['n_jobs']       = 1          # Number of jobs to run in parallel (-1 => all cores)         
default_random_params['pre_dispatch'] = '2*n_jobs' # memory is copied this many times
                                                   # reduce if you're running into memory problems
    
default_random_params['iid']          = True       # assume the folds are iid 
default_random_params['refit']        = True       # Refit the best estimator with the entire dataset 
default_random_params['cv']           = None 
default_random_params['verbose']      = 0 
#default_random_params['random_state'] = None
default_random_params['n_iter']       = 10

# set parameters for the randomized grid search
# =============================================
random_params = dict(default_random_params)

random_params['verbose']      = 1
#random_params['random_state'] = 1009
random_params['cv']           = SKFolds 
random_params['n_jobs']       = -1                # -1 => use all available cores
                                                  #       one core per fold
                                                  #       for each point in the grid

random_params['n_iter']       = 200               # choose this many random combinations of parameters
                                                  # from 'search_grid'


# perform the randomized parameter grid search
# ============================================
random_search = RandomizedSearchCV(estimator           = svc_clf, 
                                   param_distributions = search_grid, 
                                   **random_params)

random_search.fit(trainX, trainY)

pickle.dump( random_search, open( 'SVC_POLY.pkl', 'wb' ) )

print("\ntime in minutes {0:.2f}".format((time.time()-t0)/60))



