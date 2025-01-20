import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
 
import warnings
warnings.filterwarnings("ignore")

# Input: numpy array of images and number of gray levels to quantize the images down to
# Output: numpy array of images, each with only n_colors gray levels
def quantization(imgs_arr, n_colors=4):
	img_size = imgs_arr[0].shape
	res = []
	
	for img in imgs_arr:
		X = img.reshape(img_size[0] * img_size[1], 1)
		km = KMeans(n_clusters=n_colors)
		km.fit(X)
		
		img_compressed = km.cluster_centers_[km.labels_]
		img_compressed = np.clip(img_compressed.astype('uint8'), 0, 255)

		res.append(img_compressed.reshape(img_size[0], img_size[1]))
	
	return np.array(res)

# Input: A path to a folder and formats of images to read
# Output: numpy array of grayscale versions of images read from input folder, and also a list of their names
def read_dir(folder, formats=(".jpg", ".png")):
	image_arrays = []
	lst = [file for file in os.listdir(folder) if file.endswith(formats)]
	for filename in lst:
		file_path = os.path.join(folder, filename)
		image = cv2.imread(file_path)
		gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		image_arrays.append(gray_image)
	return np.array(image_arrays), lst


# Input: image object (as numpy array) and the index of the wanted bin (between 0 to 9)
# Output: the height of the idx bin in pixels
def get_bar_height(image, idx):
	# Assuming the image is of the same pixel proportions as images supplied in this exercise, the following values will work
	x_pos = 70 + 40 * idx
	y_pos = 274
	while image[y_pos, x_pos] == 0: #replace 1 -> 0
		y_pos-=1
	return 274 - y_pos

# Sections c, d

def compare_hist(src_image, target):
    
    NUMBER_HEIGHT, NUMBER_WIDTH = target.shape

    #Calculate the histogram and the cumulated histogram of target
    target_histogram = cv2.calcHist([target], [0], None, [256], [0, 256]).flatten()
    target_cum_hist = np.cumsum(target_histogram)

    #Dividing the data images into small windows
    windows = np.lib.stride_tricks.sliding_window_view(src_image, (NUMBER_HEIGHT, NUMBER_WIDTH))

    for hh in range(windows.shape[0]):
        for ww in range(windows.shape[1]):
            #Calculate the histogram and the cumulated histogram of every window
            window_histogram = cv2.calcHist([windows[hh, ww]], [0], None, [256], [0, 256]).flatten()
            window_cum_hist = np.cumsum(window_histogram)
            
            #claculate the EMD            
            emd = np.sum(np.abs(target_cum_hist - window_cum_hist))
            
            if emd < 260:
                return True  

    return False  


# Sections a, b

images, names = read_dir('data')
numbers, _ = read_dir('numbers')
#----------------------------------------
#Sections e
#Quantization - Optimal number of gray levels is  3
after = quantization(images,3)

#Transform the image to white and black
DATA_HEIGHT, DATA_WIDTH = after[0].shape
for R in range(len(images)):
        for i in range(DATA_HEIGHT):
                for j in range(DATA_WIDTH):
                        if after[R][i,j]>220:
                                after[R][i,j]=255
                        else:
                                after[R][i,j]=0

#----------------------------------------
#Cropping the data images into small windows that containe only the digit
cropped_images = []

for img in images:
    cropped_img = img[110:135, 25:41]
    cropped_images.append(cropped_img)


#------------------------------------------
#Sections f, g

combined_array = []

for i in range(len(images)):
        #claculate max_student_num using the function compare_hist
        for p in range(len(numbers)):
                if compare_hist(cropped_images[i],numbers[len(numbers)-1-p]):
                        max_student_num=len(numbers)-1-p
                        break
        #calculating  bin-height and max-bin-height using the array "after" which we got after transform the image to white and black
        bar_height = np.empty(10)
        for j in range(10):
                bar_height[j]=get_bar_height(after[i], j)
        max_bin_height=max(bar_height)

        #Final calculations : students-per-bin = round(max-student-num * bin-height / max-bin-height)
        num_students=[]
        for i in range(10):
                if max_bin_height != 0:
                        num_students.append(round(max_student_num * bar_height[i] / max_bin_height))
                else:
                        num_students.append(0)
        combined_array.append(num_students)


        
# The following print line is what you should use when printing out the final result - the text version of each histogram, basically.
#Printing
image_names = ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg", "f.jpg", "g.jpg" ]
for i in range(len(images)):
        print(f'Histogram {image_names[i]} gave {combined_array[i]}')

cv2.waitKey(0)
cv2.destroyAllWindows() 
exit()


