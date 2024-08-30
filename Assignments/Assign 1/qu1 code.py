import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imread
from scipy.ndimage import gaussian_filter
from scipy import signal as sig

"""Note:
1. Value of k lies in range(0.04, 0.06)
2. When |R| is small, which happens when λ1 and λ2 are small, the region is flat.
3. When R<0, which happens when λ1>>λ2 or vice versa, the region is edge.
4. When R is large, which happens when λ1 and λ2 are large and λ1~λ2, the region is a corner.

#### Step 2 : Harris Corner Detection
"""

def convolve2D(image, kernel):
    height, width = image.shape
    k_height, k_width = kernel.shape
    pad_height, pad_width = k_height // 2, k_width // 2

    # Pad the image
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    # Convolution operation
    output = np.zeros_like(image)
    for i in range(height):
        for j in range(width):
            output[i, j] = np.sum(padded_image[i:i + k_height, j:j + k_width] * kernel)

    return output

def harris_corner_detector(image, window_size=3, k=0.04):
    # Convert image to grayscale
    if len(image.shape) < 3:
        imggray = image
    else:
        imggray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Compute derivatives using Sobel operator
    kernel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    Ix = sig.convolve2d(imggray, kernel_x, mode='same')
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    Iy = sig.convolve2d(imggray, kernel_y, mode='same')

    ## Using in-built function
    #Ix = cv2.Sobel(imggray, cv2.CV_64F, 1, 0, ksize=3)
    #Iy = cv2.Sobel(imggray, cv2.CV_64F, 0, 1, ksize=3)

    # Compute elements of the Harris matrix M
    Ix2 = Ix ** 2
    Iy2 = Iy ** 2
    Ixy = Ix * Iy

    # Compute sums of the elements in the window
    Sxx = gaussian_filter(Ix2, window_size)
    Syy = gaussian_filter(Iy2, window_size)
    Sxy = gaussian_filter(Ixy, window_size)

    # Compute the determinant and trace of M for each pixel
    det_M = (Sxx * Syy) - (Sxy ** 2)
    trace_M = Sxx + Syy

    # Compute corner response R
    R = det_M - k * (trace_M ** 2)
    print(type(R))
    print(np.max(R), np.min(R), np.mean(R), np.median(R))
    print(R)

    # Threshold Corner Response: Apply thresholding to find corners
    #corners = np.zeros_like(R)
    #corners[R > threshold] = 255

    return R

"""#### Step 3 : Plot the corners"""

def harris_corners(image, threshold, **kwargs):
    img_copy_for_corners = np.copy(image)
    corner_response = harris_corner_detector(image, **kwargs)

    for rowindex, response in enumerate(corner_response):
        for colindex, r in enumerate(response):
            if r > threshold:
                # this is a corner, assign red color
                #img_copy_for_corners[rowindex, colindex] = [255,0,0]  # Red color in grayscale
                img_copy_for_corners[rowindex, colindex] = 255

    return img_copy_for_corners

def compare_results(image, threshold, block_size, ksize, **kwargs):
    # Apply Harris Corner detection
    custom_corners = harris_corners(image, threshold, **kwargs)
    plt.subplot(1, 2, 1)
    plt.imshow(custom_corners, cmap='bwr_r')
    plt.title('Custom Harris Corner Detection', fontsize=11)

    # Using OpenCV's corner detection
    ## Convert the input image to grayscale
    if len(image.shape) < 3:
        gray_image = image
    else:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    cv_corners = cv2.cornerHarris(gray_image, blockSize=block_size, ksize=ksize, k=0.04)
    plt.subplot(1, 2, 2)
    plt.imshow(cv_corners, cmap='bwr_r')
    plt.title('OpenCV Harris Corner Detection', fontsize=11)

    plt.show()

"""#### Step 4 : Comparison

##### Img 1
"""

dir = 'Assignment 1/Question 1/'

# Read the image
image = imread(dir+'1.gif')

img_copy_for_corners = np.copy(image)
img_copy_for_edges = np.copy(image)

custom_corners = harris_corner_detector(image,2)

for rowindex, response in enumerate(custom_corners):
    for colindex, r in enumerate(response):
        if r > 20:
            # this is a corner
            img_copy_for_corners[rowindex, colindex] = [255,0,0]
        elif r < 0:
            # this is an edge
            img_copy_for_edges[rowindex, colindex] = [0,255,0]

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,10))
ax[0].set_title("corners found")
ax[0].imshow(img_copy_for_corners)
ax[1].set_title("edges found")
ax[1].imshow(img_copy_for_edges)
plt.show()

compare_results(image, 15000, block_size=15, ksize=5)

"""##### Img 10"""

# Read the image
image = imread(dir+'10.jpg')
compare_results(image, 0, block_size=10, ksize=3)

"""##### Img 11"""

# Read the image
image = imread(dir+'11.jpg')
compare_results(image, 1000000, window_size=2, block_size=10, ksize=5)

"""##### Img 2"""

# Read the image
image = imread(dir+'2.jpeg')
compare_results(image, 400000000, window_size=2, block_size=10, ksize=5)

"""##### Img 3"""

# Read the image
image = imread(dir+'3.png')
compare_results(image, 40000000000, window_size=3, block_size=10, ksize=5)

"""##### Img 5"""

# Read the image
image = imread(dir+'5.jpg')
compare_results(image, 400000000, window_size=2, block_size=5, ksize=5)

"""##### Img 6"""

# Read the image
image = imread(dir+'6.jpg')
compare_results(image, 400000000, window_size=3, block_size=3, ksize=3)

"""##### Img 7"""

# Read the image
image = imread(dir+'7.jpg')
compare_results(image, 40000000, window_size=3, block_size=5, ksize=5)

"""##### Img 8"""

# Read the image
image = imread(dir+'8.jpg')
compare_results(image, 5000000000, window_size=1, block_size=6, ksize=5)

"""##### Img 9"""

# Read the image
image = imread(dir+'9.jpeg')
compare_results(image, 400000000, window_size=5, block_size=5, ksize=5)

# Path to the folder containing images
folder_path = 'Assignment 1/Question 1'

# Iterate through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(('.jpeg', '.jpg', '.gif', '.png')):
        image_path = os.path.join(folder_path, filename)
        print(f"Processing image: {filename}")
        compare_results(image_path)



