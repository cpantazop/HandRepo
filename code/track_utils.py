# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# !wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
# STEP 2: Create an HandLandmarker object.
# INPUT: numpy image RETURNS: detection_results 
# Image of right hand

def track_landmarks_from_img(numpy_image):       
    base_options = python.BaseOptions(model_asset_path='models/hand_landmarker.task')
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