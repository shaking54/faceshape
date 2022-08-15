import mediapipe as mp
import cv2 
import dlib
import numpy as np
import math
import os


mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
model_selection=1, min_detection_confidence=0.5)

predictor_path = "model/shape_predictor_68_face_landmarks.dat"
# Create the haar cascade  
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  

# create the landmark predictor  
predictor = dlib.shape_predictor(predictor_path)  

def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int):
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px

def get_embedding(image, face_detection):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # convert the image to grayscale  

    result = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # print("Found {0} faces!".format(len(result.detections)))  
  
    face = (result.detections[0].location_data.relative_bounding_box.xmin, 
            result.detections[0].location_data.relative_bounding_box.ymin,
            result.detections[0].location_data.relative_bounding_box.width,
            result.detections[0].location_data.relative_bounding_box.height)
    
    x,y = _normalized_to_pixel_coordinates(face[0], face[1], image.shape[1], image.shape[0])
    w,h = _normalized_to_pixel_coordinates(face[0] + face[2], face[1] + face[3], image.shape[1], image.shape[0])
    
    # Converting the OpenCV rectangle coordinates to Dlib rectangle  
    dlib_rect = dlib.rectangle(x,y,w,h)  

    detected_landmarks = predictor(image, dlib_rect).parts()
    t = np.array([[p.x, p.y] for p in detected_landmarks])
      
    return t.flatten()

def load_data(path):
  X, y = [],[]
  train_path = []
  label = {"Heart": 0, "Oblong": 1, "Oval": 2, "Round": 3, "Square": 4}

  mp_face_mesh = mp.solutions.face_mesh

  face_mesh = mp_face_mesh.FaceMesh(
      static_image_mode=True,
      max_num_faces=1,
      refine_landmarks=True,
      min_detection_confidence=0.5)

  for shape in os.listdir(path):
    print(shape, path)
    if not shape.endswith(".ini"):
      count = 0
      filepath = os.path.join(path, shape)
      for filename in os.listdir(filepath):
        try:

          image = cv2.imread(os.path.join(filepath, filename))
          results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
          face_landmarks = results.multi_face_landmarks[0]
          face = np.array([[res.x, res.y, res.z] for res in face_landmarks.landmark]).flatten()
          train_path.append(os.path.join(filepath, filename))
          X.append(face)
          y.append(label[shape])

        except:
          print(filename)
          
  return X, y, train_path


if __name__ == '__main__':
    X_train, y_train,train_path = load_data('data/FaceShape Dataset/training_set')
    X_test, y_test, test_path = load_data('data/FaceShape Dataset/testing_set')

    print(len(X_train) + len(X_test))