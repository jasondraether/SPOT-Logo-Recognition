import numpy as np
import os
from PIL import Image
import colorsys
from scipy.signal import convolve2d
from skimage.transform import resize

class MatchedFilter(object):

    def __init__(self, data_directory=''):

        self.n_rows = 300 # Height of image
        self.n_cols = 200 # Width of image
        self.n_subfilters = 4 # Downsample by 2, 4 times (so we have main filter and 4 more)
        self.max_filter_rows = self.n_rows - 4 # Since we use 'valid' only
        self.max_filter_cols = self.n_cols - 4
        self.min_filter_rows = 20
        self.min_filter_cols = 20
        self.filter_x_stride = 50
        self.filter_y_stride = 20
        self.filter_row_growth = 1 # No iea what these are for
        self.filter_col_growth = 1
        self.n_colors = 3 # Should we try grayscale?
        self.downscale_factor_init = 1 # start by applying full high res fitler
        self.downscale_growth_factor = 2
        self.class_activations = []


        # Find minimum data for each class
        n_data_min = np.inf
        self.classes = os.listdir(data_directory)
        self.n_classes = len(self.classes)
        for class_name in self.classes:
            n_data_class = len(os.listdir(os.path.join(data_directory,class_name)))
            n_data_min = n_data_class if n_data_class < n_data_min else n_data_min
        self.n_per_class = n_data_min

        self.n_data = self.n_per_class*self.n_classes

        self.labels = np.zeros((self.n_data),dtype='int')
        self.data = np.zeros((self.n_rows,self.n_cols,self.n_colors,self.n_data),dtype='uint8')

        # Read data into data arrays and labels
        for class_id in range(self.n_classes):
            class_folder = self.classes[class_id]
            image_fnames = os.listdir(os.path.join(data_directory,class_folder))
            self.labels[class_id:class_id+self.n_per_class] = class_id
            starting_index = class_id * self.n_per_class
            for datum in range(self.n_per_class):
                image_rgb = Image.open(os.path.join(data_directory,class_folder,image_fnames[datum]))
                image_hsv = image_rgb.convert(mode='HSV')
                self.data[:,:,:,starting_index+datum] = np.asarray(image_hsv,dtype='uint8')

        self.data = np.float64(self.data)
        print("Size of data: {0}".format(self.data.shape))

    def smooth_image(self):
        pass # You need to do this eventually, use a gaussian kernel

    def calculate_gradient(self):
        # Calculate gradient (G = sqrt(Gx^2 + Gy^2)) using sobel operator, use valid only
        difference_filter = np.asarray([1, 0, -1])
        average_filter = np.asarray([1, 2, 1])

        self.gradient = np.zeros((self.n_rows-2, self.n_cols-2, self.n_colors, self.n_data))
        self.row_convolution = np.zeros((self.n_rows, self.n_cols-2, 2*self.n_colors, self.n_data))
        self.split_gradient = np.zeros((self.n_rows-2, self.n_cols-2, 2*self.n_colors, self.n_data))

        for col in range(2, self.n_cols):
            self.row_convolution[:,col-2,0:3,:] = self.data[:,col,:,:] - self.data[:,col-2,:,:]
            self.row_convolution[:,col-2,3:6,:] = self.data[:,col,:,:] + 2*self.data[:,col-1,:,:] + self.data[:,col-2,:,:]

        for row in range(2, self.n_rows):
            self.split_gradient[row-2,:,0:3,:] = self.row_convolution[row,:,0:3,:] + 2*self.row_convolution[row-1,:,0:3,:] + self.row_convolution[row-2,:,0:3,:]
            self.split_gradient[row-2,:,3:6,:] = self.row_convolution[row,:,3:6,:] - self.row_convolution[row-2,:,3:6,:]

        for row in range(0, self.n_rows-2):
            for col in range(0, self.n_cols-2):
                self.gradient[row,col,0,:] = np.sqrt(self.split_gradient[row,col,0,:]**2 + self.split_gradient[row,col,3,:]**2)
                self.gradient[row,col,1,:] = np.sqrt(self.split_gradient[row,col,1,:]**2 + self.split_gradient[row,col,4,:]**2)
                self.gradient[row,col,2,:] = np.sqrt(self.split_gradient[row,col,2,:]**2 + self.split_gradient[row,col,5,:]**2)


        # test = np.uint8(self.gradient[:,:,:,0])
        # gradient_image = Image.fromarray(test,'HSV')
        # gradient_image.show()


    def create_filters(self):
        self.image_filters = [] # List of plain image filters, at highest resolution
        self.gradient_filters = [] # List of image gradient filters, at highest resolution

        image_filter = np.zeros((self.max_filter_rows,self.max_filter_cols,self.n_colors))
        gradient_filter = np.zeros((self.max_filter_rows,self.max_filter_cols,self.n_colors))

        for class_id in range(self.n_classes):
            starting_index = class_id * self.n_per_class
            image_filter = np.zeros((self.max_filter_rows,self.max_filter_cols,self.n_colors))
            gradient_filter = np.zeros((self.max_filter_rows,self.max_filter_cols,self.n_colors))
            for row in range(2, self.n_rows-2):
                for col in range(2, self.n_cols-2):
                    image_filter[row-2,col-2,:] = np.mean(self.data[row,col,:,starting_index:starting_index+self.n_per_class],axis=1) # Later try using gradient
                    gradient_filter[row-2,col-2,:] = np.mean(self.gradient[row,col,:,starting_index:starting_index+self.n_per_class],axis=1) # Later try using gradient

            image_filter = image_filter[::-1,:,:] # flip left to right
            image_filter = image_filter[:,::-1,:] # flip top to bottom

            gradient_filter = gradient_filter[::-1,:,:] # flip left to right
            gradient_filter = gradient_filter[:,::-1,:] # flip top to bottom

            # test = np.uint8(image_filter)
            # plain_image = Image.fromarray(test,'HSV')
            # plain_image.show()
            #
            # test = np.uint8(gradient_filter)
            # gradient_image = Image.fromarray(test,'HSV')
            # gradient_image.show()

            self.image_filters.append(image_filter)
            self.gradient_filters.append(gradient_filter)

        print("Filters created")

    def apply_filters(self, test_image):

        print("Applying filters")

        # Check for shape mismatch
        assert test_image.shape[0] == self.n_rows
        assert test_image.shape[1] == self.n_cols
        assert test_image.shape[2] == self.n_colors
        assert test_image.dtype == 'float64'


        n_test_rows = test_image.shape[0]
        n_test_cols = test_image.shape[1]


        for class_id in range(self.n_classes):

            activations = []
            # Apply main filters for class
            image_filter = self.image_filters[class_id] # Probably not the best way to do this but whatever
            gradient_filter = self.gradient_filters[class_id]

            image_filter = image_filter[:,::-1,:] # flip top to bottom
            image_filter = image_filter[::-1,:,:] # flip left to right

            gradient_filter = gradient_filter[:,::-1,:] # flip top to bottom
            gradient_filter = gradient_filter[::-1,:,:] # flip left to right

            # Apply subfilters for class
            downscale_factor = self.downscale_factor_init
            subfilter_rows = self.max_filter_rows
            subfilter_cols = self.max_filter_cols
            for subfilter_num in range(self.n_subfilters):
                if subfilter_rows >= self.min_filter_rows and subfilter_cols >= self.min_filter_cols:
                    subfilter = resize(image_filter,(subfilter_rows,subfilter_cols) ,anti_aliasing=True)
                    averaging_factor = subfilter_rows*subfilter_cols
                    # test = np.uint8(subfilter)
                    # plain_image = Image.fromarray(test,'HSV')
                    # plain_image.show()

                    # Upper left coordinates
                    for row in range(0, n_test_rows,self.filter_y_stride):
                        for col in range(0, n_test_cols, self.filter_x_stride):
                            x0 = col
                            x1 = col + subfilter_cols
                            y0 = row
                            y1 = row + subfilter_rows

                            if y1 < n_test_rows and x1 < n_test_cols:
                                activation = np.zeros((subfilter_rows,subfilter_cols,self.n_colors))
                                for sub_row in range(subfilter_rows):
                                    for color in range(self.n_colors):
                                        activation[sub_row,:,color] += np.dot(subfilter[sub_row,:,color], test_image[row+sub_row,x0:x1,color])
                                # TODO: store activation somewhere? probably in a list
                                activation /= (averaging_factor)
                                filename = 'test-{0}-{1}-{2}-{3}.txt'.format(class_id,subfilter_num,row,col)
                                # np.savetxt(filename,activation)
                                activations.append((activation, x0, x1, y0, y1))



                    subfilter_rows //= downscale_factor
                    subfilter_cols //= downscale_factor
                    downscale_factor *= self.downscale_growth_factor
                else:
                    print("Subfilter rows too small!")

            # test = np.uint8(activations[0])
            # # print(test)
            # image = Image.fromarray(test,'HSV')
            # image.show()
            self.class_activations.append(activations)

        for class_id in range(self.n_classes):
            activations = self.class_activations[class_id]
            for i in range(len(activations)):
                print("Class {0} activation with value {1}\n".format(self.classes[class_id], np.mean(activations[i][0])))


            # Well this doesnt work because activations looks broke yo
            values = [val[0] for val in activations]
            values_array = np.asarray(values)
            print(values_array.shape)
            #print(values_array)
            values_array_averaged = np.mean(values_array,axis=2)
            print(values_array_averaged.shape)
            index = np.argmax(values_array_averaged)
            print("Max:", activations[index])

    # def apply_filters(self):
    #     '''
    #     Need to apply filter (not much rotation needed, maybe 15 degrees each way?) across entire image,
    #     gradually make filter bigger until it is the size of the original image (may be good idea to have filter at high
    #     resolution, and downsample as needed)
    #
    #     Then, take average arg max of each heat point
    #     '''
    #     pass


def main():
    obj = MatchedFilter(data_directory='filters/')
    obj.calculate_gradient()
    obj.create_filters()

    image_rgb = Image.open('test_images/test.png')
    image_hsv = image_rgb.convert(mode='HSV')
    image_hsv = np.float64(image_hsv)

    obj.apply_filters(test_image=image_hsv)

if __name__ == '__main__':
    main()
