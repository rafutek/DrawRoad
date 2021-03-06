**DrawRoad_py** contains several python files:

- *saveRoadLines.py* detects every line in each frame of the video road_car_view.mp4, and asks the user to save it or not. The user might choose the good detected lines, so they are saved in goodLines.txt.

- *displaySavedLines.py* displays the lines saved in goodLines.txt in a graph, as two groups, the left/red lines and the right/blue lines. This can be used by the user to choose the left and right line intervals in *drawRoad.py*.

- *drawRoad.py* detects the left and right line on each frame if there are, and draw in real-time the associated road as a map (in top view). 


The detection of the lines has several states:
- read frame from the video
- cut it in order to have only a road image
- apply HSV conversion and mask to reveal the road lines
- find the lines with Hough transform
- select a good left and right line thanks to the intervals
