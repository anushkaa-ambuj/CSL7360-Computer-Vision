import cv2
import numpy as np

dir = 'Assignment 1/Question 3/'

# Load the images
image1 = cv2.imread(dir + '000000.png', 0)  # Load image in grayscale
image2 = cv2.imread(dir + '000023.png', 0)  # Load image in grayscale

# Read Fundamental matrix from file
with open(dir + 'FM.txt', 'r') as file:
    content = file.read()

# Extract numerical values from the content
values_str = content.replace('F=', '').strip().replace('[', '').replace(']', '').replace(';', '').replace('\n', ',').split(',')
#print(values_str)
values = [float(val.strip()) for val in values_str]

# Reshape the array into a 3x3 matrix
F = np.array(values).reshape((3, 3))
print("Fundamental Matrix \n",F)

def compute_epipolar_line(F, point, image_num):
    if image_num == 1:
        return np.dot(F, point)
    elif image_num == 2:
        return np.dot(np.transpose(F), point)

def draw_line(image, line):
    x0, y0 = map(int, [0, -line[2] / line[1]])
    x1, y1 = map(int, [image.shape[1], -(line[2] + line[0] * image.shape[1]) / line[1]])
    cv2.line(image, (x0, y0), (x1, y1), (0, 255, 0), 1)

# Choose a point on the epipolar line
point = np.array([10, 10, 1])

# Compute epipolar lines for both images
line1 = compute_epipolar_line(F, point, 1)
line2 = compute_epipolar_line(F, point, 2)

# Draw epipolar lines on the images
image1_with_lines = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
image2_with_lines = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
draw_line(image1_with_lines, line1)
draw_line(image2_with_lines, line2)

# Display the images with epipolar lines
cv2.imshow('Image 1 with Epipolar Lines', image1_with_lines)
cv2.imshow('Image 2 with Epipolar Lines', image2_with_lines)
cv2.waitKey(0)
cv2.destroyAllWindows()