import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,ClassifierMixin
from scipy.spatial.distance import cdist

class kNN(BaseEstimator, ClassifierMixin):
  def __init__(self, n_neighbors:int = 3):
    self.n_neighbors = n_neighbors
    self.points = None
    self.labels = None

  def fit(self, X, y):
    # self.set_params(X_train = X, y_train = y)
    self.points = np.copy(X)
    self.ylabels = np.copy(y)
    return self

  def predict(self, X):
    dist_mat = cdist(X, self.points)
    k_neighbors = np.argpartition(dist_mat, kth=self.n_neighbors)[ : ,  : self.n_neighbors]
    predictions = None
    k_neighbors_labels = np.array(self.ylabels)[k_neighbors.astype(int)]
    predictions = np.sign(np.array(np.sum(k_neighbors_labels, axis=1)))
    predictions[predictions == 0] = 1
    return predictions