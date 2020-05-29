import numpy as np
import os
from PIL import Image
import colorsys

class MatchedFilter(object):

    def __init__(self, data_directory=''):

        self.n_rows = 200
        self.n_cols = 300
        self.max_filter_rows = self.n_rows
        self.max_filter_cols = self.n_cols
        self.min_filter_rows = 20
        self.min_filter_cols = 20
        self.filter_x_stride = 1
        self.filter_y_stride = 1
        self.filter_row_growth = 1
        self.filter_col_growth = 1
        self.n_colors = 3 # Should we try grayscale?

        # Find minimum data for each class
        n_data_min = np.inf
        classes = os.listdir(data_directory)
        self.n_classes = len(classes)
        for class in classes:
            n_data_class = os.listdir(os.path.join(data_directory,class))
            n_data_min = n_data_class if n_data_class < n_data_min else n_data_min
        self.n_per_class = n_data_min

        self.n_data = self.n_per_class*self.n_classes

        self.labels = np.zeros((self.ndata),dtype='int')
        self.data = np.zeros((self.nrows,self.ncols,self.ncolors,self.ndata),dtype='float64')

        # Read data into data arrays and labels
        for class_id in range(self.n_classes):
            class_folder = classes[class_id]
            image_fnames = os.listdir(os.path.join(data_directory,class_folder))
            self.labels[class_id:class_id+self.n_per_class] = class_id
            starting_index = class_id * self.n_per_class
            for datum in range(self.n_per_class):
                image_rgb = Image.open(os.path.join(data_directory,class_folder,image_fnames[datum]))
                image_hsv = image_rgb.convert(mode='HSV')
                self.data[:,:,:,starting_index+datum] = np.asarray(image_hsv)

    def smooth_image(self):
        pass


    def create_filters(self):
        self.filters = [] # List of filters, at highest resolution

        for class_id in range(self.n_classes):
            filter = np.zeros((self.max_filter_rows,self.max_filter_cols))
            starting_index = class_id * self.n_per_class
            filter = np.mean
            for datum in range(self.n_per_class):
                image_rgb = self.data[:,:,:,starting_index+datum]

    def apply_filters(self):
        '''
        Need to apply filter (not much rotation needed, maybe 15 degrees each way?) across entire image,
        gradually make filter bigger until it is the size of the original image (may be good idea to have filter at high
        resolution, and downsample as needed)

        Then, take average arg max of each heat point
        '''
        pass
