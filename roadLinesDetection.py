
# We want to:
# detect the lines of a road in a video
# filter lines to get the two of the roads only
# display the road as a map thanks to these lines

############## imports ##############

import cv2 
import numpy as np
import matplotlib.pyplot as plt



############## variables definition ##############

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

window_name = 'Line Detection'


############## functions definition ##############

def lineAnalysisAndDisplay(line, img):
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)



# 'ESC' to return false, 'SPACE' to next frame, any other to next line
def linesAnalysisAndDisplay(lines,window, img):
        if lines is not None:
            for line in lines:
                lineAnalysisAndDisplay(line, img)
        
        continueProgram = True

        cv2.imshow(window, img)
        c = cv2.waitKey(delay)
        if c == 27:
            continueProgram = False

        return continueProgram




############## main program ##############

videoName = "road_car_view.mp4"
video = cv2.VideoCapture(videoName)

if not video.isOpened():
    print("Error opening video "+videoName)

while(video.isOpened()):
    read_flag, frame = video.read()
    # if the video is finished, we play it again
    if not read_flag:
        print("play video again")
        video = cv2.VideoCapture(videoName)
        continue
    # we only want the lower part of the road
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
    # display lines and save lines user wants
    if not linesAnalysisAndDisplay(lines, window_name, img_road):
        break

video.release()
cv2.destroyAllWindows()
