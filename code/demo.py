print("Hello")
import track_utils
import display_utils
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('images/example.png')
# reads an image in the BGR format
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# print(type(img))
results = track_utils.track_landmarks_from_img(img)
print(results.hand_landmarks[0][0].x)

imgShow = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imshow('image',imgShow)
cv2.waitKey(0)
cv2.destroyAllWindows()

annotated_image = display_utils.draw_landmarks_on_image(img, results)
annotated_imageShow = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
cv2.imshow('tracked',annotated_imageShow)
cv2.waitKey(0)
cv2.destroyAllWindows()