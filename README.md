**DrawRoad_py** contains several python files:

- *saveRoadLines.py* detect every line in each frame of the video road_car_view.mp4, and ask the user to save it or not. The user might to choose the good detected lines, so they are saved in goodLines.txt.

- *displaySavedLines.py* display the lines saved in goodLines.txt in a graph, as two groups, the left/red lines and the right/blue lines. This can be used by the user to choose the left and right lines interval in *drawRoad.py*.

- *drawRoad.py* detect the left and right line on each frame if there are, and draw the associated road as a map (in top view). 


The detection of the lines has several states:
- read frame from the video
- cut it in order to have only a road image
- apply HSV conversion and mask to reveal the road lines
- find the lines with Hough transform
- select a good left and right line thanks to the intervals
