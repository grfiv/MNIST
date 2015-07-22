#Support Vector Machines for classification
I built the model with IPython notebooks using scikit-learn's **SVC**. These notebooks contain examples of:

* using **StandardScaler** to find the column mean and std of the training set and applying it to both the training and test sets.

* using **PCA** to effect dimensionality reduction

* a heatmap of the random grid search results for **C** and **gamma** vs. accuracy (*very* helpful)

* learning curves for the best fitted model found with the grid search (also helpful)
* 
* validation curves for **C** and **gamma** (not sure what to make of them)

* colorful **confusion matrix plot** of the test-set results(merely interesting)

I tried two kernels:  

* **RBF**
* **Polynomial**

I experimented with using twice as much training data (combining the original and deskewed datasets) and I experimented using PCA to reduce dimensionality.


