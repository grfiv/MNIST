#scikit-learn Gradient Boosted Trees
R's **gbm** is a memory pig: fitting 2,000 trees on 122Gb blew up with an out-of-memory condition. scikit is much more frugal. I built the model with IPython notebooks:

* *gbm.scikit_gridsearch.ipynb* uses **GridSearchCV** to find the optimal **max_depth**, which is the number of terminal nodes in each tree, reflecting the extent of predictor interaction in the target function.

* *gbm.scikit_curves.ipynb* uses cross validation to find the optimal **n_estimators**, which is the number of trees to fit (it is also an example of the use of explicit Python parallelization with **multiprocessing**)

* *gbm.scikit_benchmark.ipynb* uses the parameters found above and fits the full MNIST training set and predicts the full MNIST test set

* *gbm.scikit_kaggle.ipynb* runs the Kaggle data through the model

Based on Ridgeway's R vignette I made the assumption that the smaller the **learning_rate** the better the fit and that 0.01 would require 10x the number of trees required by 0.1. But I used 0.1 in my estimation of **max_depth** on the assumption that the level of interaction would be the same, simply because the model takes hours and hours to run as it is. Based on Friedman's work I made the additional assumption that **subsample < 1** would be beneficial; I failed to test whether **max_features < 1** would be beneficial, again simply for expedience.
