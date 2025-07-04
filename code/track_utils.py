import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# !wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

# INPUT: numpy image RETURNS: detection_results 
def track_landmarks_from_img(numpy_image):    
    # choose options and create detector   
    base_options = python.BaseOptions(model_asset_path='../models/hand_landmarker.task')
    VisionRunningMode = vision.RunningMode
    options = vision.HandLandmarkerOptions(base_options=base_options,running_mode=VisionRunningMode.IMAGE, num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)
    
    # Load the input image from a file.
    # mp_image = mp.Image.create_from_file("images/example.png")
    # Load the input image from a numpy array.
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)
    
    #Detect hand landmarks from the input image.
    detection_result = detector.detect(mp_image)
    return(detection_result)

def get_real_landmarks(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # print(handedness[idx].display_name)

    IsLeft = handedness[idx].display_name == "Left"

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [int(landmark.x * width) for landmark in hand_landmarks]
    y_coordinates = [int(landmark.y * height) for landmark in hand_landmarks]
    z_coordinates = [int(landmark.z * width) for landmark in hand_landmarks]
   
  return x_coordinates,y_coordinates,z_coordinates,height,width,IsLeft