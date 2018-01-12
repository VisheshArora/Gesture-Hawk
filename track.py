import cv2
import numpy as np

def nothing(x):
    pass
cv2.namedWindow('image')
img = cv2.VideoCapture(0)
cv2.createTrackbar('l_H','image',110,255,nothing)
cv2.createTrackbar('l_S','image',50,255,nothing)
cv2.createTrackbar('l_V','image',50,255,nothing)
cv2.createTrackbar('h_H','image',130,255,nothing)
cv2.createTrackbar('h_S','image',255,255,nothing)
cv2.createTrackbar('h_V','image',255,255,nothing)
while(1):
   _,frame = img.read()
   
   hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
   
   lH = cv2.getTrackbarPos('l_H','image')
   lS = cv2.getTrackbarPos('l_S','image')
   lV = cv2.getTrackbarPos('l_V','image')
   hH = cv2.getTrackbarPos('h_H','image')
   hS = cv2.getTrackbarPos('h_S','image')
   hV = cv2.getTrackbarPos('h_V','image')
   lower_R = np.array([lH,lS,lV])
   higher_R = np.array([hH,hS,hV])
   
   mask = cv2.inRange(hsv, lower_R, higher_R)
   
   res = cv2.bitwise_and(frame,frame, mask= mask)
   cv2.imshow('image',res)
   
   






   k = cv2.waitKey(1) & 0xFF
   if k == 27:
        break   

cv2.destroyAllWindows()
