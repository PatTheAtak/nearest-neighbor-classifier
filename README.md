# nearest-neighbor-classifier
## Numpy Implementation Of The Nearest Neighbor Classifier

When building a machine learning model, you need to implement two functions:
1. Train Function
   - This function takes in the training features and corresponding training labels and gets you model to memorize some patterns and representations found in the data.
2. Predict Function
   - This function takes the model that wastrained on the training data and makes some predictions from the use of a test dataset.

The **K-Nearest Neighbor Classifier** is an example of such a machine learning model. It is a simple algorithm that stores all available cases and classifies new cases based on a similarity measure. A data point is classified by a majority vote of its neighbors, with the data point being assigned to the class most common amongst its K nearest neighbors measured by a distance function. If K = 1, then the data point is simply assigned to the class of its nearest neighbor. 

