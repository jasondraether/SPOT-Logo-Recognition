import numpy as np
import os
from PIL import Image
from scipy import signal

class MatchedFilter(object):

    '''
    Set necessary parameters and load data and labels into their proper arrays
    '''
    def __init__(self, data_directory='data', color_space='HSV'):

        self.data_directory = data_directory
        self.color_space = color_space

        self.n_image_rows = 300 # Image height
        self.n_image_cols = 200 # Image width
        if color_space == 'grayscale':
            self.n_colors = 1
        else:
            self.n_colors = 3

        self.n_subfilters = 4 # We downsample by 2 for each new subfilter
        self.max_filter_rows = self.n_image_rows - 4 # Use 'valid' only
        self.max_filter_cols = self.n_image_cols - 4
        self.min_filter_rows = 25 # Arbitrary, any smaller and probably too low resolution
        self.min_filter_cols = 25
        self.filter_row_stride = 20 # How far to advance sliding window during filtering
        self.filter_col_stride = 50
        self.downscale_growth_factor = 2 # Downscale by multiples of 2 each new subfilter

        self.class_activations = [] # Store list of list of activations for each class

        # Find minimum number of images for each class, and choose minimum of that
        self.n_data_min = np.inf
        self.classes = os.listdir(self.data_directory) # Classes determined based on filesystem
        self.n_classes = len(self.classes)
        for class_name in self.classes:
            class_datapath = os.path.join(self.data_directory,class_name)
            class_files = os.listdir(class_datapath)
            n_data = len(class_files)
            self.n_data_min = n_data if n_data < self.n_data_min else self.n_data_min

        # Calculate total number of data samples
        self.n_data = self.n_data_min*self.n_classes
        self.n_data_per_class = self.n_data_min

        # Load data and labels
        self.labels = np.zeros((self.n_data),dtype='int')
        self.data = np.zeros((self.n_image_rows,self.n_image_cols,self.n_colors,self.n_data),dtype='float64')
        for class_id in range(self.n_classes):
            class_name = self.classes[class_id]
            class_datapath = os.path.join(self.data_directory,class_name)
            class_files = os.listdir(class_datapath)
            starting_index = class_id * self.n_data_per_class
            self.labels[starting_index:starting_index+self.n_data_per_class] = class_id
            for datum in range(self.n_data_per_class):
                datum_filename = class_files[datum]
                datum_datapath = os.path.join(class_datapath,datum_filename)
                image_rgb = Image.open(datum_datapath)
                image_converted = image_rgb.convert(mode=self.color_space)
                image_array = np.asarray(image_converted,dtype='uint8') # For some reason PIL reads image as uint8
                self.data[:,:,:,starting_index+datum] = image_array / 255.0 # Load into data array and scale from 0-1.0 (float64)

    '''
    Apply Gaussian kernel to image
    '''
    def smooth_image(self, kernel_size=5, std_dev=1):
        self.data_gaussian = np.zeros((self.n_image_rows,self.n_image_cols,self.n_colors,self.n_data),dtype='float64')
        gaussian_kernel = signal.gaussian(kernel_size,std_dev)
        gaussian_kernel_2d = np.outer(gaussian_kernel,gaussian_kernel)
        normalizing_factor = np.sum(gaussian_kernel_2d) # Will flatten array
        gaussian_kernel_2d /= normalizing_factor
        for datum in range(self.n_data):
            for color in range(self.n_colors):
                self.data_gaussian[:,:,color,datum] = signal.convolve2d(self.data[:,:,color,datum],gaussian_kernel_2d,mode='same')

            img = Image.fromarray(np.uint8(self.data_gaussian[:,:,:,datum]*255.0),mode='HSV')
            img = img.convert(mode='RGB')
            img.save('convolution-{0}.jpg'.format(datum))


def main():
    obj = MatchedFilter(data_directory='filters/')
    obj.smooth_image()



if __name__ == '__main__':
    main()
