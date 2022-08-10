import cv2
import mediapipe as mp
import numpy as np
import os
import sys

import matplotlib.pyplot as plt

from PIL import Image 

from skimage.feature import hog
from skimage import data, exposure

def meshExtractor(path, savepath, hist=False):

  phases = ['training_set','testing_set']

  mp_drawing = mp.solutions.drawing_utils
  mp_drawing_styles = mp.solutions.drawing_styles
  mp_face_mesh = mp.solutions.face_mesh

  face_mesh = mp_face_mesh.FaceMesh(
      static_image_mode=True,
      max_num_faces=1,
      refine_landmarks=True,
      min_detection_confidence=0.5)

  if not os.path.exists(savepath):
    os.mkdir(savepath)
  for phase in phases:
    dirs = os.path.join(path, phase)
    if not phase.endswith('.ini'):
      if not os.path.exists(savepath+phase):
        os.mkdir(savepath+phase)
      for shape in os.listdir(dirs):
        if not shape.endswith('.ini'):
          shapepath = os.path.join(dirs, shape)
          for filename in os.listdir(shapepath):
            try:
              file = os.path.join(shapepath, filename)
              p = savepath+phase+"/"+shape+"/"
              if not os.path.exists(p):
                os.mkdir(p)
              spath = os.path.join(p, filename)
              image = cv2.imread(file)
              # Convert the BGR image to RGB before processing.
              results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
              # Print and draw face mesh landmarks on the image.
              if not results.multi_face_landmarks:
                continue
              face_landmarks = results.multi_face_landmarks[0]

              if not hist:
                annotated_image = np.zeros(image.shape)
              
                for face_landmarks in results.multi_face_landmarks:
                  mp_drawing.draw_landmarks(
                      image=annotated_image,
                      landmark_list=face_landmarks,
                      connections=mp_face_mesh.FACEMESH_TESSELATION,
                      landmark_drawing_spec=None,
                      connection_drawing_spec=mp_drawing_styles
                      .get_default_face_mesh_tesselation_style())

                  h, w, c = annotated_image.shape
                  cx_min=  w
                  cy_min = h
                  cx_max= cy_max= 0
                  for id, lm in enumerate(face_landmarks.landmark):
                      cx, cy = int(lm.x * w), int(lm.y * h)
                      if cx<cx_min:
                          cx_min=cx
                      if cy<cy_min:
                          cy_min=cy
                      if cx>cx_max:
                          cx_max=cx
                      if cy>cy_max:
                          cy_max=cy
                    
                cv2.imwrite(spath, annotated_image[int(0.9*cy_min):int(cy_max*1.05),int(0.9*cx_min):int(cx_max*1.05)])
              
              else:
                
                annotated_image = image.copy()
                
                for face_landmarks in results.multi_face_landmarks:
                  h, w, c = image.shape
                  cx_min=  w
                  cy_min = h
                  cx_max= cy_max= 0
                  for id, lm in enumerate(face_landmarks.landmark):
                      cx, cy = int(lm.x * w), int(lm.y * h)
                      if cx<cx_min:
                          cx_min=cx
                      if cy<cy_min:
                          cy_min=cy
                      if cx>cx_max:
                          cx_max=cx
                      if cy>cy_max:
                          cy_max=cy
                # abc = image[int(0.9*cy_min):int(cy_max*1.05),int(0.9*cx_min):int(cx_max*1.05)]
                fd, hog_image = hog(image, 
                                orientations=8, pixels_per_cell=(16, 16),
                                cells_per_block=(1, 1), visualize=True)
                hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
                hog_image_rescaled = np.float32(hog_image_rescaled)
                hog_image_rescaled = cv2.cvtColor(hog_image_rescaled, cv2.COLOR_GRAY2RGB)
                # print(hog_image_rescaled)
                mp_drawing.draw_landmarks(
                      image=hog_image_rescaled,
                      landmark_list=face_landmarks,
                      connections=mp_face_mesh.FACEMESH_TESSELATION,
                      landmark_drawing_spec=None,
                      connection_drawing_spec=mp_drawing_styles
                      .get_default_face_mesh_tesselation_style())
                
                hog_image_rescaled = cv2.convertScaleAbs(hog_image_rescaled, alpha=(255.0))

                #plt.imsave(spath, hog_image[int(0.9*cy_min):int(cy_max*1.05),int(0.9*cx_min):int(cx_max*1.05)])
                cv2.imwrite(spath, hog_image_rescaled[int(0.9*cy_min):int(cy_max*1.05),int(0.9*cx_min):int(cx_max*1.05)])
            except:
              print(filename)


if __name__ == '__main__':

    path = 'data/FaceShape Dataset/'
    savepath = 'data/FaceMeshHog/'
    meshExtractor(path, savepath, hist=True)