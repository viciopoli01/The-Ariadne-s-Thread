import cv2
import numpy as np

# Load the image
image = cv2.imread('/home/viciopoli/datasets/MarsNeRF/sol449/test_1/0009.png')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow('Original Image', gray)
# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

cv2.imshow('blurred', blurred)
# Use adaptive thresholding to segment the rocks
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 5)

cv2.imshow('thr', thresh)
# Perform morphological operations to clean up the segmentation
kernel = np.ones((3, 3), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

# Find contours of segmented objects
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
result = image.copy()
cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

# Display the result
cv2.imshow('Segmented Rocks', result)
cv2.waitKey(0)
cv2.destroyAllWindows()