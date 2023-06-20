# This code's functionality is largely from outside sources and is filled with comments which act as annotations 
# to describe what certain lines do and what I learned while working on this project.

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

def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    # If there are no lines to draw, exit.
    if lines is None:
        return
    # Make a copy of the original image.
    img = np.copy(img)
    # Create a blank image that matches the original in size.
    line_img = np.zeros(
        (
            img.shape[0],
            img.shape[1],
            3
        ),
        dtype=np.uint8,
    )
    # Loop over all lines and draw them on the blank image.
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    # Merge the image with the lines onto the original.
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    # Return the modified image.
    return img

image = mpimg.imread('testimage.jpg') # read in our original image

height, width, color = image.shape # image.shape is run through the CV2 library and returns a tuple of dimensions (height, weight, color) we use this notation to 'unpack' the tuple and assign each value to a variable.

# Defining our vertices for the triangle we are interested in (0,h) maps to the bottom left corner, (w,h) maps to the bottom right corner, and the third points maps to the middle of the picture
region_of_interest_vertices = [
    (0, height),
    (width / 2, height / 2),
    (width, height),
] 

#plt.imshow(image) # Creates a figure and draws an image on the figure
#plt.show() # Displays our image!

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

#plt.figure()
#plt.imshow(cropped_image)
#plt.show()

# Cool! We now have all the edges of our lane lines. However, these edges are all just a bunch of pixels next to each other. We need a way to link pixels together to create a set of lines rather than groups of pixels.
# After the Canny algorithm, the image consists of mostly "blank" pixels and a couple colored pixels. We need to find some way to mathematically relate the colored pixels together and define lines.
# We can utilize a linear algebra feature called the Hough Transform to simplify our problem and attain only one line.
# Disclaimer: This next section of comments is all about the Hough Transform and the math behind it.
# I find this kind of math really cool so I decided to document it but if you're focused more on the practicality, feel free to skip ahead.
# Skip to line 80 if you don't want to read more about Hough Transforms.

# This explanation is going to require a little familiarity with linear algebra.
# Give this video a watch to see visual diagrams and a great explanation of this concept: https://www.youtube.com/watch?v=4zHbI-fFIlI&ab_channel=ThalesSehnK%C3%B6rting
# Imagine a line in the x-y plane. It can be represented by the classic linear equation y = ax + b.
# In this equation, y and x are variables, representing a large array of values(points) that sit on the line. a and b are constants, defining the angle and direction of the line.
# Let's define two points on the line: (xi, yi) and (xj, yj). These points are two pairs of x and y values assigned to a specific spot in the x-y plane that happen to intersect with our line.
# Now, let's invert our equation, identifying it in terms of a and b in the a-b plane. This turns our equation y = ax + b into:
# b = -xia + yi and b = -xja + yj
# Since xi and xj are different values, their respective lines have different slopes in the a-b plane. Since yi and yj have different values, their lines have different y-intercepts in the a-b plane. This creates two fundamentally different lines.
# Since the lines have different slopes, they are not parallel. This means that at some point in the a-b plane, the two lines will intersect.
# So now, let's find the intersecting point of these lines. First, let's rewrite our line equations into general form.
# xia + b - yi = 0 and xja + b - yj = 0. Assume (a,b) is the point of intersection
# Now, using cross multiplication, we get the following relation: (a/((1*-yj)-(1*-yi))) = (-b/((xi*-yj)-(xj*-yi))) = (1/((xi*1)-(xj*1)))
# Solving for a, we get a = ((1*-yj)-(1*-yi))/((xi*1)-(xj*1)) = (-yj+yi)/(xi-yj)
# Solving for b, we get b = -((xi*-yj)-(xj*-yi))/((xi*1)-(xj*1)) = ((xj*-yi)-(xi*-yj))/((xj*1)-(xi*1)) = (-xj*yi+xi*yj)/(xj-xi)
# So, we can represent our intersection point (a,b) as ((-yj+yi)/(xi-yj), (-xj*yi+xi*yj)/(xj-xi))
# Now this obviously represents a point in our a-b plane, or Hough Space. But, what's interesting is that this point, when inverted back into the x-y plane, or "Regular Space", it represent our initial line, y = ax + b.
# What does this mean for us? This shows us that lines in regular space become points in Hough space and points in Hough space become lines in regular space.

# Ok! The math is over. If you skipped the huge section of comments above, here's a quick TLDR.
# The Hough Transform utilizes Hough Space, a plane that uses the a and b from y = ax + b as its axes instead of x and y.
# We will consider the x-y plane as Regular Space.
# Points in regular space (x,y) are represented as lines in Hough space (b = -xa + y).
# Lines in regular space are represented as points in Hough space.
# So, our current image includes a ton of potential, or "candidate", lines in regular space. If we represent these in Hough space, each line turns into a singular point.
# Every potential line emanating from a single pixel can be represented as a point in Hough space. Now, we just need to look for the one point in common between every pixel in Hough space.
# Bringing that point back into regular space, we get a single line that passes through all the pixels.
# Luckily, OpenCV does all this complex math for us.

lines = cv2.HoughLinesP( # Built-in OpenCV method
    cropped_image,
    rho=5, # Distance Resolution in pixels - Changed from 6 to 5.
    theta=np.pi / 60, # Angle Resolution in radians
    threshold=160, # Accumulator threshold - determines the necessary "votes" to display a line
    lines=np.array([]), # Output vector
    minLineLength=40,
    maxLineGap=25
)
# print(lines)

# Awesome, Python now outputs the two endpoints of each detected line! Let's overlay these lines onto our original image to see if they match up with the lanes in the image.
# I have added the draw_lines function above.

line_image = draw_lines(image, lines) #  We will call our function here.
plt.figure()
plt.imshow(line_image)
plt.show()

# Great! The lane lines are now highlighted. But, the left lane is broken up since the lines are only highlighted where it is white. For the purpose of lane detection, we want to link these broken white dashes.
