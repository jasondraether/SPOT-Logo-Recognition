import numpy as np
import os
from PIL import Image
from scipy import signal
from skimage.transform import resize
from matplotlib.image import imread

debug_directory = 'debug_images/'
debug = 1

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
        self.downscale_factor_init = 1

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
    def smooth_image(self, std_dev=1.0): # Can play around with different values
        self.data_gaussian = np.zeros((self.n_image_rows,self.n_image_cols,self.n_colors,self.n_data),dtype='float64')
        kernel_size = np.ceil(6.0*std_dev) # Based off wikipedia, kernel should be ceil(6.0*std_dev) each dimension
        gaussian_kernel = signal.gaussian(kernel_size,std_dev)
        gaussian_kernel_2d = np.outer(gaussian_kernel,gaussian_kernel)
        normalizing_factor = np.sum(gaussian_kernel_2d) # Will flatten array
        gaussian_kernel_2d /= normalizing_factor
        for datum in range(self.n_data):
            for color in range(self.n_colors):
                self.data_gaussian[:,:,color,datum] = signal.convolve2d(self.data[:,:,color,datum],gaussian_kernel_2d,mode='same')

            if debug:
                img = Image.fromarray(np.uint8(self.data_gaussian[:,:,:,datum]*255.0),mode='HSV')
                img = img.convert(mode='RGB')
                img.save(os.path.join(debug_directory, 'gaussian_filtered-{0}.jpg'.format(datum)))

        self.data = self.data_gaussian # Replace data with gaussian data

    def calculate_gradient(self):
        # Difference filter: [1, 0, -1]
        # Average filter: [1, 2, 1]
        # Calculate gradient as G = sqrt(Gx^2 + Gy^2) based on sobel operator, valid only
        self.gradient = np.zeros((self.n_image_rows-2,self.n_image_cols-2,self.n_colors,self.n_data))
        # Since we separate the convolution, 0:3 holds difference and 3:6 holds average for row convolution
        self.row_convolution = np.zeros((self.n_image_rows,self.n_image_cols-2,2*self.n_colors,self.n_data))
        # 0:3 holds Gx, 3:6 holds Gy for gradient components
        self.gradient_components = np.zeros((self.n_image_rows-2,self.n_image_cols-2,2*self.n_colors,self.n_data))

        # It's a convolution, so the filtering will actually be backward
        for col in range(2, self.n_image_cols):
            self.row_convolution[:,col-2,0:3,:] = self.data[:,col,:,:] - self.data[:,col-2,:,:]
            self.row_convolution[:,col-2,3:6,:] = self.data[:,col,:,:] + 2*self.data[:,col-1,:,:] + self.data[:,col-2,:,:]

        for row in range(2, self.n_image_rows):
            self.gradient_components[row-2,:,0:3,:] = self.row_convolution[row,:,0:3,:] + 2*self.row_convolution[row-1,:,0:3,:] + self.row_convolution[row-2,:,0:3,:]
            self.gradient_components[row-2,:,3:6,:] = self.row_convolution[row,:,3:6,:] - self.row_convolution[row-2,:,3:6,:]

        # Combine the two directions
        for row in range(0,self.n_image_rows-2):
            for col in range(0,self.n_image_cols-2):
                self.gradient[row,col,0,:] = np.sqrt(self.gradient_components[row,col,0,:]**2 + self.gradient_components[row,col,3,:]**2)
                self.gradient[row,col,1,:] = np.sqrt(self.gradient_components[row,col,1,:]**2 + self.gradient_components[row,col,4,:]**2)
                self.gradient[row,col,2,:] = np.sqrt(self.gradient_components[row,col,2,:]**2 + self.gradient_components[row,col,5,:]**2)

        if debug:
            for datum in range(self.n_data):
                # Combined gradient
                img = Image.fromarray(np.uint8(self.gradient[:,:,:,datum]*255.0),mode='HSV')
                img = img.convert(mode='RGB')
                img.save(os.path.join(debug_directory, 'gradient-{0}.jpg'.format(datum)))
                # Gx
                img = Image.fromarray(np.uint8(self.gradient_components[:,:,0:3,datum]*255.0),mode='HSV')
                img = img.convert(mode='RGB')
                img.save(os.path.join(debug_directory, 'gradient_x-{0}.jpg'.format(datum)))
                # Gy
                img = Image.fromarray(np.uint8(self.gradient_components[:,:,3:6,datum]*255.0),mode='HSV')
                img = img.convert(mode='RGB')
                img.save(os.path.join(debug_directory, 'gradient_y-{0}.jpg'.format(datum)))

    def create_filters(self):
        self.image_filters = [] # Filters based on image data
        self.gradient_filters = [] # Filters based on gradients

        # Get gradient and normal filter for each class
        # Cut off first and last rows/columns, average values for each pixel and color,
        # flip over y-axis and then over x-axis
        for class_id in range(self.n_classes):
            starting_index = class_id * self.n_data_per_class
            image_filter = np.zeros((self.max_filter_rows,self.max_filter_cols,self.n_colors))
            gradient_filter = np.zeros((self.max_filter_rows,self.max_filter_cols,self.n_colors))
            for row in range(2,self.n_image_rows-2):
                for col in range(2,self.n_image_cols-2):
                    for color in range(self.n_colors):
                        image_filter[row-2,col-2,color] = np.sum(self.data[row,col,color,starting_index:starting_index+self.n_data_per_class])/self.n_data_per_class
                        gradient_filter[row-2,col-2,color] = np.sum(self.gradient[row,col,color,starting_index:starting_index+self.n_data_per_class])/self.n_data_per_class

            image_filter = image_filter[::-1,:,:] # Flip left-to-right
            image_filter = image_filter[:,::-1,:] # Flip top-to-bottom
            gradient_filter = gradient_filter[::-1,:,:]
            gradient_filter = gradient_filter[:,::-1,:]

            self.image_filters.append(image_filter)
            self.gradient_filters.append(gradient_filter)

            if debug:
                img = Image.fromarray(np.uint8(image_filter*255.0),mode='HSV')
                img = img.convert(mode='RGB')
                img.save(os.path.join(debug_directory, 'image_filter-{0}.jpg'.format(class_id)))

                img = Image.fromarray(np.uint8(gradient_filter*255.0),mode='HSV')
                img = img.convert(mode='RGB')
                img.save(os.path.join(debug_directory, 'gradient_filter-{0}.jpg'.format(class_id)))

    def apply_filters(self, test_image):

        test_rows = test_image.shape[0]
        test_cols = test_image.shape[1]
        test_colors = test_image.shape[2]

        for class_id in range(self.n_classes):
            # Can probably compute number of new filters at runtime but I'm lazy...
            activations = []

            # Grab filters
            image_filter = self.image_filters[class_id]
            gradient_filter = self.gradient_filters[class_id]

            # Flip them bottom-to-top, then right-to-left
            image_filter = image_filter[:,::-1,:]
            image_filter = image_filter[::-1,:,:]
            gradient_filter = gradient_filter[:,::-1,:]
            gradient_filter = gradient_filter[::-1,:,:]

            # Generate subfilters
            downscale_factor = self.downscale_factor_init # Starts at 1
            subfilter_rows = self.max_filter_rows
            subfilter_cols = self.max_filter_cols
            for subfilter_id in range(self.n_subfilters):
                # Make sure subfilter isn't too small
                if subfilter_rows >= self.min_filter_rows and subfilter_cols >= self.min_filter_cols:
                    subfilter = resize(image_filter,(subfilter_rows,subfilter_cols),anti_aliasing=True) # We already smoothed image so we might not need anti-aliasing
                    averaging_factor = subfilter_rows*subfilter_cols*self.n_colors

                    # Slide subfilter around image
                    for y0 in range(0, test_rows, self.filter_row_stride):
                        for x0 in range(0, test_cols, self.filter_col_stride):
                            # Four coordinates of (y0,y1,x0,x1) form sliding window
                            y1 = y0 + subfilter_rows
                            x1 = x0 + subfilter_cols

                            # Check that we're in bounds
                            if y1 < test_rows and x1 < test_cols:
                                activation = np.zeros((subfilter_rows,subfilter_cols,self.n_colors))
                                for color in range(self.n_colors):
                                    activation[:,:,color] = (np.subtract(subfilter[:,:,color], test_image[y0:y1,x0:x1,color]))**2

                                activation_sum = np.sum(activation)/averaging_factor
                                activations.append((activation_sum,y0,x0,y1,x1))

                                if debug:
                                    img = Image.fromarray(np.uint8(activation*255.0),mode='HSV')
                                    img = img.convert(mode='RGB')
                                    img.save(os.path.join(debug_directory, 'activation-{0}-{1}-{2}-{3}-{4}-{5}.jpg'.format(class_id, subfilter_id, y0, x0, y1, x1)))

                    subfilter_rows //= downscale_factor
                    subfilter_cols //= downscale_factor
                    downscale_factor *= self.downscale_growth_factor

            self.class_activations.append(activations)

    def inference(self):

        min_activation = np.inf
        coordinates = (0,0,0,0)
        match_id = -1
        for class_id in range(self.n_classes):

            activations = self.class_activations[class_id]

            n_activations = len(activations)
            for activation_id in range(n_activations):

                activation_data = activations[activation_id]
                print(class_id, activation_data)
                value = activation_data[0]
                if value < min_activation:
                    min_activation = value
                    coordinates = activation_data[1:5]
                    match_id = class_id

        print("Max at coordinates: {0} {1} {2} {3} with class {4}".format(coordinates[0],coordinates[1],coordinates[2],coordinates[3],self.classes[match_id]))
        return coordinates

def main():
    obj = MatchedFilter(data_directory='filters/')
    obj.smooth_image()
    obj.calculate_gradient()
    obj.create_filters()

    test_path = 'test_images/test.png'

    image_rgb = Image.open(test_path)
    image_hsv = image_rgb.convert(mode='HSV')
    image_hsv = np.float64(image_hsv)

    obj.apply_filters(test_image=image_hsv) # TODO: pass in image array
    (y0,x0,y1,x1) = obj.inference()

    if debug:

        image_rgb = Image.open(test_path)
        image_converted = image_rgb.convert(mode='HSV')
        img = np.array(image_converted,dtype='uint8') # For some reason PIL reads image as uint8
        img = imread(test_path)
        print(img.shape)
        for row in range(y0,y1):
            img[row,x0,0:3] = np.asarray([1.0,1.0,1.0])
            img[row,x1,0:3] = np.asarray([1.0,1.0,1.0])
        for col in range(x0,x1):
            img[y0,col,0:3] = np.asarray([1.0,1.0,1.0])
            img[y1,col,0:3] = np.asarray([1.0,1.0,1.0])

        img = Image.fromarray(np.uint8(img*255.0),mode='HSV')
        img = img.convert(mode='RGB')

        img.save(os.path.join(debug_directory,'match_location.jpg'))


if __name__ == '__main__':
    main()
