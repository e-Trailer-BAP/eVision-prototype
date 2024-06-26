import cv2
import numpy as np
import math

# To read the image
image = cv2.imread('data/calib_images/left-undistort.png')

# Specify 4 corners in the image that need perspective transform
pt0 = [817, 668] #bottom right
pt1 = [348,528] # bottom left
pt2 = [519,299] # top left
pt3 = [850,371] # top right

offset=23
br=[70,77+offset]
bl=[-100,137+offset]
tl=[-50,297+offset]
tr=[120,237+offset]

camx=500
camy=500
scale=1

# Specify the width and the height of transformed image
width, height = math.floor(scale*camx*2), math.floor(scale*camy)

# Width and height of transformed image (Choice)
pts1 = np.float32([pt0, pt1, pt2, pt3])
pts2 = np.float32([[scale*(camx+br[0]), scale*(camy-br[1])],
                   [scale*(camx+bl[0]), scale*(camy-bl[1])],
                   [scale*(camx+tl[0]), scale*(camy-tl[1])],
                   [scale*(camx+tr[0]), scale*(camy-tr[1])]])
print(pts2)

#bottom right, bottom left, top left, top right

# Apply perspective transform Method
matrix = cv2.getPerspectiveTransform(pts1, pts2)
result = cv2.warpPerspective(image, matrix, (width, height))

print(matrix)

# Show the result
print('pre transform')

cv2.imshow("pre",image)

print('post transform')
cv2.imshow("post",result)
cv2.waitKey()
