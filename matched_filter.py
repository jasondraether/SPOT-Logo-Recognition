import numpy as np
import os
from PIL import Image

class MatchedFilter(object):

    def __init__(self, data_directory=''):

        self.n_rows = 200
        self.n_cols = 300
        self.n_filter_rows = 20
        self.n_filter_cols = 20
        self.n_colors = 3 # Should we try grayscale?

        # Find minimum data for each class
        n_data_min = np.inf
        classes = os.listdir(data_directory)
        self.n_classes = len(classes)
        for class in classes:
            n_data_class = os.listdir(data_directory+class)
            n_data_min = n_data_class if n_data_class < n_data_min else n_data_min
        self.n_per_class = n_data_min

        self.n_data = self.n_per_class*self.n_classes

        self.labels = np.zeros((self.ndata),dtype='int')
        self.data = np.zeros((self.nrows,self.ncols,self.ncolors,self.ndata),dtype='float64')


    def create_filters(self):


    def apply
