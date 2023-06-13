import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Setting up a python function to alter an image and define a region to analyze our image
def region_of_interest(img, vertices):
    mask = np.zeros_like(img) # zeroes_like returns an array of zeroes with the same size as the input array (img)
    match_mask_color = 255 # Sets our mask color to black
    
    cv2.fillPoly(mask, vertices, match_mask_color) # OpenCV function used to draw filled polygons in the mask color
    masked_image = cv2.bitwise_and(img, mask) # bitwise_and combines two matrices of 0s and 1s and returns a 1 when both matrices have a 1 at that point and returns a 0 anywhere else
    return masked_image

image = mpimg.imread('testimage.jpg') # read in our original image

height, width, color = image.shape # image.shape is run through the CV2 library and returns a tuple of dimensions (height, weight, color) we use this notation to 'unpack' the tuple and assign each value to a variable.

# Defining our vertices for the triangle we are interested in (0,h) maps to the bottom left corner, (w,h) maps to the bottom right corner, and the third points maps to the middle of the picture
region_of_interest_vertices = [
    (0, height),
    (width / 2, height / 2),
    (width, height),
] 

plt.imshow(image) # Creates a figure and draws an image on the figure
plt.show() # Displays our image!

# Now, we will begin our edge detection process. In mathematical terms, an edge can be defined as an area where pixel colors change drastically (in our case, the white of the lane lines changes to the gray of the road) To achieve this, we are looking at the differences in intensity in the pixels. Thus, we don't need to consider the color of the pixels, so we can convert our image to grayscale with a cv2 method to eliminate unnecessary data
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Now, the mathematical problem of detecting an edge is definitely solvable and luckily, an algorithm for solving it has already been developed - the power of the CS community :). Essentially, the algorithm utilizes gradients which are a multivariable calculus topic (a class I haven't taken yet whoops). From my understanding (aka googling) gradients are derivatives fr multivariable functions. They help describe how a functiont that depends on multiple independent variables changes
cannyed_image = cv2.Canny(gray_image, 100, 200)
# The next step in this process was to apply a Canny Operator to our now greyscale image but I had no idea what Canny's detection algorithm did or how it worked. So, after watching a couple very informational YouTube videos (shout out Dr. Shree Nayar at Columbia), I'm prepared to break down how this algorithm works.
# Step 1: Apply a Gaussian Blur to smooth the image
# Step 2: Apply the Sobel Operator to compute the image gradient -> this gives us the derivative in the x-direction and in the y-direction at each pixel
# Step 3: Using the derivatives found above, calculate the gradient magnitude at each pixel (magnitude increases with pixel brightness)
# Step 4: Find the orientation of the gradient at each pixel
# Step 5: Apply a one-direction Laplacian along the gradient direction/orientation -> Canny uses 1D to eliminate the possiblity of other data in different directions interfering with our edge definition
# Step 6: Find strong zero crossings in Laplacian to find the edge location

# We can finally crop our image with our intended vertex region and retrieve a final image that highlights our lane edges!
cropped_image = region_of_interest(
    cannyed_image,
    np.array([region_of_interest_vertices], np.int32)
)

plt.figure()
plt.imshow(cropped_image)
plt.show()