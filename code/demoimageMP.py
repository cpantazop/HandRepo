print("Hello")
import track_utils
import display_utils
import cv2
import numpy as np
from matplotlib import pyplot as plt

# reads an image in the BGR format and converts it to RGB
img = cv2.imread('images/example_right.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# print(type(img))
# gets the landmark results from mediapipe and given image
results = track_utils.track_landmarks_from_img(img)
# print(results.hand_landmarks[0][0].x) #prints the x value of the first keypoint
# print(results) #prints the x value of the first keypoint
# Extract the x, y, and z values into a list of lists
landmarks = [[lm.x, lm.y, lm.z] for lm in results.hand_landmarks[0][:]]
print(landmarks)

# displays original image
imgShow = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imshow('image',imgShow)
cv2.waitKey(0)
cv2.destroyAllWindows()
#creates and displays annotated image
annotated_image = display_utils.draw_landmarks_on_image(img, results)
annotated_imageShow = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
cv2.imshow('tracked',annotated_imageShow)
cv2.waitKey(0)
cv2.destroyAllWindows()