import numpy as np
import os
from PIL import Image
import colorsys
from scipy.signal import convolve2d
from skimage.transform import resize

class MatchedFilter(object):

    def __init__(self, data_directory=''):

        self.n_rows = 200
        self.n_cols = 300
        self.n_subfilters = 4 # Downsample by 2, 4 times (so we have main filter and 4 more)
        self.max_filter_rows = self.n_rows - 4 # Since we use 'valid' only
        self.max_filter_cols = self.n_cols - 4
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
        pass # You need to do this eventually, use a gaussian kernel

    def calculate_gradient(self):
        # Calculate gradient (G = sqrt(Gx^2 + Gy^2)) using sobel operator, use valid only
        difference_filter = np.asarray([1, 0, -1])
        average_filter = np.asarray([1, 2, 1])

        self.gradient = np.zeros((self.n_rows-2, self.n_cols-2, self.n_colors, self.n_data))
        self.row_convolution = np.zeros((self.n_rows-2, self.n_cols-2, 2*self.n_colors, self.n_data))
        self.split_gradient = np.zeros((self.n_rows-2, self.n_cols-2, 2*self.n_colors, self.n_data))

        for col in range(2, self.n_cols):
            self.row_convolution[:,col-2,0:3,:] = np.dot(self.data[:,col-2:col+1,0:3,:], difference_filter) # Double check this
            self.row_convolution[:,col-2,3:6,:] = np.dot(self.data[:,col-2:col+1,3:6,:], average_filter) # Double check this

        for row in range(2, self.n_rows):
            self.split_gradient[row-2,:,0:3,:] = np.dot(self.row_convolution[row-2:row+1,:,0:3,:], average_filter)
            self.split_gradient[row-2,:,3:6,:] = np.dot(self.row_convolution[row-2:row+1,:,3:6,:], difference_filter)

        for row in range(0, self.n_rows-2):
            for col in range(0, self.n_cols-2):
                self.gradient[row,col,0,:] = np.sqrt(self.split_gradient[row,col,0,:]**2 + self.split_gradient[row,col,3,:]**2)
                self.gradient[row,col,1,:] = np.sqrt(self.split_gradient[row,col,1,:]**2 + self.split_gradient[row,col,4,:]**2)
                self.gradient[row,col,2,:] = np.sqrt(self.split_gradient[row,col,2,:]**2 + self.split_gradient[row,col,5,:]**2)



    def create_filters(self):
        self.image_filters = [] # List of plain image filters, at highest resolution
        self.gradient_filters = [] # List of image gradient filters, at highest resolution

        image_filter = np.zeros((self.max_filter_rows,self.max_filter_cols,self.n_colors))
        gradient_filter = np.zeros((self.max_filter_rows,self.max_filter_cols,self.n_colors))

        for class_id in range(self.n_classes):
            starting_index = class_id * self.n_per_class
            image_filter = np.zeros((self.max_filter_rows,self.max_filter_cols,self.n_colors))
            gradient_filter = np.zeros((self.max_filter_rows,self.max_filter_cols,self.n_colors))
            for row in range(2, self.nrows-2):
                for col in range(2, self.ncols-2):
                    image_filter[row,col,:] = np.mean(self.data[row,col,:,starting_index:starting_index+self.n_per_class],axis=1) # Later try using gradient
                    gradient_filter[row,col,:] = np.mean(self.gradient[row,col,:,starting_index:starting_index+self.n_per_class],axis=1) # Later try using gradient

            image_filter = image_filter[::-1,:,:] # flip left to right
            image_filter = image_filter[:,::-1,:] # flip top to bottom

            gradient_filter = gradient_filter[::-1,:,:] # flip left to right
            gradient_filter = gradient_filter[:,::-1,:] # flip top to bottom

            self.image_filters.append(image_filter)
            self.gradient_filters.append(gradient_filter)

    def apply_filters(self):

        downscale_factor = 2

        for class_id in range(self.n_classes):

            # Apply main filters for class

            # Apply subfilters for class
            subfilter_rows = self.max_filter_rows / downscale_factor
            subfilter_cols = self.max_filter_cols / downscale_factor
            for subfilter_num in range(n_subfilters):
                subfilter = np.zeros((subfilter_rows, subfilter_cols, self.n_colors))

                # Apply subfilter

                subfilter_rows /= downscale_factor
                subfilter_cols /= downscale_factor
                downscale_factor *= 2


    def apply_filters(self):
        '''
        Need to apply filter (not much rotation needed, maybe 15 degrees each way?) across entire image,
        gradually make filter bigger until it is the size of the original image (may be good idea to have filter at high
        resolution, and downsample as needed)

        Then, take average arg max of each heat point
        '''
        pass
