# We want to:
# get the lines saved in the file
# display them in a graph in function of their category

############## imports ##############

import numpy as np
import matplotlib.pyplot as plt


############## variables definition ##############

left_lines = np.empty((0,2), int)
right_lines = np.empty((0,2), int)

############## functions definition ##############


############## main program ##############

filename = "./goodLines.txt"
lines = np.genfromtxt(filename)

left_lines = np.vstack((left_lines, lines[lines[:,0] < 0]))
right_lines = np.vstack((right_lines, lines[lines[:,0] > 0]))


plt.title("Seperated Lines") 
plt.xlabel("rho") 
plt.ylabel("theta") 
plt.plot(left_lines[:,0],left_lines[:,1], "ro") 
plt.plot(right_lines[:,0],right_lines[:,1], "b^") 
plt.show()