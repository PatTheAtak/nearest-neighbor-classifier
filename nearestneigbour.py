import numpy as np

# Nearest neighbour algorithm
class NearestNeighbour:
    def __init__(self):
        pass

    # Train method memorizes the training data
    def train(self, X, y):  # X is N x D, y is N x 1 (one-hot encoded)
        """X is N x D where each row is an example. Y is 1-dimension of size N"""
        # the nearest neighbour classifier simply remembers all the training data
        self.Xtr = X
        self.ytr = y

    # Predict method predicts the label of an example X
    def predict(self, X):  #
        """X is N x D where each row is an example we wish to predict label for"""
        num_test = X.shape[0]

        # lets make sure that the output type matches the input type
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)

        # loop over all test rows
        for i in range(num_test):
            # find the nearest training image to the i'th test image
            # using the L1 distance (sum of absolute value differences)
            distances = np.sum(np.abs(self.Xtr - X[i, :]), axis=1)
            min_index = np.argmin(distances)

        return Ypred
