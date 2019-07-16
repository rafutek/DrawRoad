
# We want to:
# detect the lines of a road in a video
# filter lines to get the two of the roads only
# display the road as a map thanks to these lines

############## imports ##############

import cv2 
import numpy as np
import matplotlib.pyplot as plt



############## variables definition ##############
window_original_road = 'Lines Detection'
window_drawn_road = 'Road Drawing'


delay = 200

max_value = 255

low_H = 0
high_H = max_value

low_S = 0
high_S = 50

low_V = 105
high_V = max_value

low_edge = 100
high_edge = 200


hThreshold = 40

min_rho_left_line = -300
max_rho_left_line = -200
min_theta_left_line = 2
max_theta_left_line = 3

min_rho_right_line = 50
max_rho_right_line = 150
min_theta_right_line = 0
max_theta_right_line = 1.2


size = 200, 100, 1
img = np.zeros(size, dtype=np.uint8)
line_length = 5
last_x = int(img.shape[1]/2)
last_y = int(img.shape[0])

theta_straight_line = 1.45
degree_straight_line = -90
coef = degree_straight_line/theta_straight_line

############## functions definition ##############

# draw the line (rho, theta) in img with color
def drawLine(rho, theta, img, color):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(img,(x1,y1),(x2,y2),color,2)

# computes and return the next line point with the first point, the angle and the length of the line
def nextLinePoint(x1,y1, angle, length):
    x2 =  int(round(x1 + length * np.cos(angle * np.pi / 180.0)))
    y2 =  int(round(y1 + length * np.sin(angle * np.pi / 180.0)))
    return x2, y2

# create an array of zeros of the size passed in parameter
def createArray(height, width, depth):
    size = height, width, depth
    return np.zeros(size, dtype=np.uint8)

# creates a new array/image if the point (x2,y2) is not in the array img 
# this new array/image contain the previous img at the right place (below, to the left, etc..) in order to have (x2,y2) at a border
# and change the values of the points (x1,y1) and (x2,y2)
def newArray(x1, y1, x2, y2, img):
    new_array = img

    if not nextPointDrawable(x2,y2,img):
        if x2 < 0:
            space = abs(x2)
            space_array = createArray(img.shape[0], space, 1)
            new_array = np.append(space_array, img, axis=1)           
            x2 += space
            x1 += space

        elif x2 > img.shape[1]:
            space = x2 - img.shape[1]
            space_array = createArray(img.shape[0], space, 1)
            new_array = np.append(img, space_array, axis=1)

        
        if y2 < 0:
            space = abs(y2)
            space_array = createArray(space, img.shape[1], 1)
            new_array = np.append(space_array, img, axis=0)           
            y2 += space
            y1 += space
        elif y2 > img.shape[0]:
            space = y2 - img.shape[0]
            space_array = createArray(space, img.shape[1], 1)
            new_array = np.append(img, space_array, axis=0)   

    
    return new_array, x1, y1, x2, y2

# say if the point (x2,y2) is contained the array/image or not
def nextPointDrawable(x2,y2, img):
    return x2 >= 0 and x2 < img.shape[1] and y2 >= 0 and y2 < img.shape[0]

# analyse the line passed in parameter to say if it's a left line, a right line, or none of them
def lineAnalysis(rho, theta):
    left_line = False
    right_line = False 

    if rho > min_rho_left_line and rho < max_rho_left_line and theta > min_theta_left_line and theta < max_theta_left_line:
        left_line = True
    elif rho > min_rho_right_line and rho < max_rho_right_line and theta > min_theta_right_line and theta < max_theta_right_line:
        right_line = True
    
    return left_line, right_line


# take every line detected in parameter, and find one left line and one right line
def linesLeftAndRight(lines):
    left_rho = left_theta = 0
    right_rho = right_theta = 0
    left_line_found = right_line_found = False

    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            left_line, right_line = lineAnalysis(rho, theta)
            
            if left_line and not left_line_found:
                left_line_found = True
                left_rho = rho
                left_theta = theta
            elif right_line and not right_line_found:
                right_line_found = True
                right_rho = rho
                right_theta = theta
            
            if left_line_found and right_line_found:
                break

    return left_rho, left_theta, right_rho, right_theta


# return the angle in degree of the line to draw
def thetaToDegree(theta):
    return coef * theta



############## main program ##############

videoName = "road_car_view.mp4"
video = cv2.VideoCapture(videoName)

if not video.isOpened():
    print("Error opening video "+videoName)

while(video.isOpened()):
    read_flag, frame = video.read()
    # if the video is finished, we get out of the loop
    if not read_flag:
        print(videoName+" finished")
        break
    # we only want the lower part of the image: the road
    height, width = frame.shape[:2]
    img_road = frame[height-80:height, 0:width]

    # now we have only the road with the lines
    # so let's convert this image to hsv...
    hsv = cv2.cvtColor(img_road, cv2.COLOR_BGR2HSV)
    # ...threshold it in order to get only the road lines....
    low_HSV = np.array([low_H, low_S, low_V])
    up_HSV = np.array([high_H, high_S, high_V])
    hsv_mask = cv2.inRange(hsv, low_HSV, up_HSV)
    # ...and blur the final image
    hsv_blured = cv2.GaussianBlur(hsv_mask, (7, 7), 0)

    # now, detect the edges on the image
    edges = cv2.Canny(hsv_mask, low_edge, high_edge)

    # then, detect the lines of the edges
    lines = cv2.HoughLines(edges, 1, np.pi / 180, hThreshold)

    # find left line and right line
    left_rho, left_theta, right_rho, right_theta = linesLeftAndRight(lines)

    # draw
    if left_rho != 0 and left_theta != 0 and right_rho != 0 and right_theta != 0:
        drawLine(left_rho, left_theta, img_road, (0,0,255))
        drawLine(right_rho, right_theta, img_road, (255,0,0))
        
        theta_mean_line = left_theta - right_theta 
        x2, y2 = nextLinePoint(last_x, last_y, thetaToDegree(theta_mean_line), line_length)
        
        # if necessary, increase img to be able to draw the line completly
        img, last_x, last_y, x2, y2  = newArray(last_x,last_y,x2,y2,img)


    # display
    cv2.imshow(window_original_road, img_road)
    if cv2.waitKey(delay) == 27:
        break
    cv2.line(img,(last_x,last_y),(x2,y2),(255,255,255),2)
    cv2.imshow(window_drawn_road, img)

    # new point becomes last point at each iteration
    last_x = x2
    last_y = y2
    


video.release()
cv2.destroyAllWindows()
