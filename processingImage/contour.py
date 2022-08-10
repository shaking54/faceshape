import cv2
import dlib
import os
import re
# import argparse
import numpy as np

def contourExtractor(modelpath, datapath, result):
  for dir in os.listdir(datapath):
    if not dir.startswith('.') and not dir.endswith('.ini'):
      shapepath = os.path.join(datapath, dir)
      res = os.path.join(result, dir)
      print(res)
      if not os.path.exists(res):
        os.mkdir(res)
      for filename in os.listdir(shapepath):
        try:
          image_path = os.path.join(shapepath, filename)
          image = cv2.imread(image_path)
          face_detector = dlib.get_frontal_face_detector()
          dets = face_detector(image, 1)
          predictor = dlib.shape_predictor(modelpath)
          x,w,y,h= 0,0,0,0
          previous_img = None
          maxArea = 0
          for d in dets:
            shape = predictor(image, d)
            img = np.zeros(image.shape, dtype = "uint8")
            points = shape.num_parts
            prevPoint = shape.part(0)
            for i in range(1, points):
              p = shape.part(i)
              cv2.circle(img, (p.x, p.y), 2, 255, 6)
              cv2.line(img,(prevPoint.x, prevPoint.y), (p.x, p.y), 255, 4)
              prevPoint = p
            x,w,y,h = d.left(), d.top(), d.right(), d.bottom()
            area = (d.right() - d.left()) * (d.bottom() - d.top())
            if area > maxArea:
              previous_img = img
              maxArea = area
              x,w,y,h = d.left(), d.top(), d.right(), d.bottom()
          a,b = int(0.1*w), int(0.1*h)
          cv2.imwrite(os.path.join(res, filename), img[w-a:h+b, x-a:y+b])
          break
        except:
          print(filename)


if __name__ == '__main__':
  modelpath = "model/face_contour_17.dat"
  datapath = ["data/FaceShape Dataset/training_set", "data/FaceShape Dataset/testing_set"]
  result = ["data/Contour/training_set", "data/Contour/testing_set"]
  for i in range(2):
    contourExtractor(modelpath, datapath[i], result[i])