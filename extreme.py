import cv2 
import numpy as np 
import imutils
import serial
#ser = serial.Serial('/dev/ttyACM0', baudrate = 9600)       #ACM0

#ser.write(b'70')

cp1 = cv2.imread('ph1.jpg')
hsa = cv2.cvtColor(cp1, cv2.COLOR_BGR2HSV)
    
lower_blue = np.array([110,50,50])
upper_blue = np.array([130,255,255])
    
mask2 = cv2.inRange(hsa, lower_blue, upper_blue)
res2 = cv2.bitwise_and(cp1,cp1, mask= mask2)
imgrey2 = cv2.cvtColor(res2,cv2.COLOR_BGR2GRAY)
ret2,thresh2 = cv2.threshold(imgrey2,127,255,cv2.THRESH_BINARY_INV)
_,conto, hierarc = cv2.findContours(imgrey2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
sz1 = len(conto)
ar1 = np.arange(sz1).reshape((sz1,1))
for i in range(0,sz1):
 ar1[i] = cv2.contourArea(conto[i])
maxAr1 = np.amax(ar1)
cindex1 = np.argmax(ar1)
cn2 = conto[cindex1]

#leftmost = tuple(conto[conto[:,:,0].argmin()][0])
#rightmost = tuple(conto[conto[:,:,0].argmax()][0])
extRight = tuple(cn2[cn2[:, :, 0].argmax()][0])
extLeft = tuple(cn2[cn2[:, :, 0].argmin()][0])
cv2.circle(cp1, extRight, 8, (0,255,0), -1)
#ret = cv2.matchShapes(cn1,cn2,1,0.0)
#print(ret)

hull = cv2.convexHull(cn2,returnPoints = False)
defects = cv2.convexityDefects(cn2,hull)
defShape = defects.shape[0]
#print(defShape)
dis = np.arange(defShape).reshape((defShape,1))
for i in range(defects.shape[0]):
    s,e,f,d = defects[i,0]
    start = tuple(cn2[s][0])
    end = tuple(cn2[e][0])
    far = tuple(cn2[f][0])
    dis[i] = -1*((far[0]-start[0])*(start[1]-end[1])-(far[1]-start[1])*(start[0]-end[0]))/(((start[1]-end[1])**2 + (start[0] - end[0])**2)**(0.5))
    cv2.line(cp1,start,end,[0,255,0],2)
    cv2.line(cp1,far,end,[0,255,0],2)
    cv2.line(cp1,far,start,[0,255,0],2)
    cv2.circle(cp1,far,5,[0,0,255],-1)
 
cv2.drawContours(cp1, conto, cindex1, (0,255,0), 3)
maxLine_ind = np.argmax(dis)
#print(dis)
a,d,b,w = defects[maxLine_ind,0]
farEnd = tuple(cn2[b][0]) 
#slope = (-extRight[1]+farEnd[1])/(extRight[0]-farEnd[0])


distRF = ((extRight[1]-farEnd[1])**2 + (extRight[0]-farEnd[0])**2)**(0.5)
distLF = ((extLeft[1]-farEnd[1])**2 + (extLeft[0]-farEnd[0])**2)**(0.5)
if distRF<distLF:
 slope = (-extRight[1]+farEnd[1])/(extRight[0]-farEnd[0])
 print('Right Gesture')
if distRF>distLF:
 slope = (-extLeft[1]+farEnd[1])/(extLeft[0]-farEnd[0])
 print('Left Gesture')
print(slope)
cv2.circle(cp1,farEnd,5,[255,0,0],-1)
cv2.circle(cp1,(0,30),5,[255,0,0],-1)
cv2.imshow("Title",cp1)
 

cv2.waitKey()
cv2.destroyAllWindows()
